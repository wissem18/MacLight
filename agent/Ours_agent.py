import torch, numpy as np, torch.nn.functional as F

class MacLight:
    def __init__(self, policy_net, critic_net,
                 actor_lr=1e-4, critic_lr=5e-3,
                 gamma=0.9, lmbda=0.9,pred_coef=0.01, epochs=20, eps=0.2,
                 device='cpu'):

        self.actor   = policy_net.to(device)
        self.critic  = critic_net.to(device)
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma, self.lmbda, self.pred_coef = gamma, lmbda, pred_coef
        self.epochs, self.eps  = epochs, eps
        self.device            = device



    # ───────────────────────────────────────────────────────────────
    # take_action unchanged
    # ───────────────────────────────────────────────────────────────
    def take_action(self, state):
        state  = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.distributions.Categorical(self.actor(state)).sample()
        return action.item()

    
    def update(self, transition_dict, agent_name,attention,temp_enc, predictor, accumulate_attn_grad=False):
        """One agent’s PPO update with temporal encoder + dynamic predictor.  
        """
        # ------------------------------------------------------------------
        # 1. pull this agent’s local trajectory tensors
        # ------------------------------------------------------------------
        to_t = lambda x,dtype: torch.tensor(x, dtype=dtype, device=self.device)
        states      = to_t(transition_dict['states'][agent_name]     , torch.float32)
        actions     = to_t(transition_dict['actions'][agent_name]    , torch.int64 ).view(-1,1)
        rewards     = to_t(transition_dict['rewards'][agent_name]    , torch.float32).view(-1,1)
        next_states = to_t(transition_dict['next_states'][agent_name], torch.float32)
        dones       = to_t(transition_dict['dones'][agent_name]      , torch.int32 ).view(-1,1)

        # ------------------------------------------------------------------
        # 2. build whole-state tensors (T, N, 33)  for shared Attention
        # ------------------------------------------------------------------
        agt_order = list(transition_dict['states'].keys())
        idx       = agt_order.index(agent_name)
        stack = lambda key, dtype=torch.float32: torch.stack([to_t(transition_dict[key][a], dtype) for a in agt_order], dim=1)
        whole_state       = stack('states')
        whole_next_state  = stack('next_states')
        whole_action      = stack('actions',dtype=torch.int64)
        whole_reward      = stack('rewards')

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # ------------------------------------------------------------------
        # 3. PPO E epochs
        # ------------------------------------------------------------------
        for epoch in range(self.epochs):

            # 1 – spatial embeddings via GAT
            h_all, A = attention(whole_state)                                    # (T,N,d)

            # stack the last K GAT outputs for each timestep -> Transformer
            K = temp_enc.K
            pad  = torch.zeros(K - 1, *h_all.shape[1:], device=self.device)
            cat  = torch.cat([pad, h_all], dim=0)                           # (T+K-1,N,d)
            hist = torch.stack([cat[t:t + K] for t in range(h_all.size(0))], 0)
            hist = hist.permute(0, 2, 1, 3)                                 # (T,N,K,d)
            z_all = temp_enc(hist)                                          # (T,N,d)

            z_t     = z_all[:, idx, :]                                      # (T,d)  this node
            z_next  = torch.cat([z_t[1:], z_t[-1:].clone()], dim=0)

            if epoch == self.epochs - 1:
                self.full_A = A.detach().cpu().to(torch.float16)

            # 2 – critic targets
            v_now  = self.critic(states, z_t)
            v_next = self.critic(next_states, z_next).detach()
            td_tgt = rewards + self.gamma * v_next * (1 - dones)
            td_del = td_tgt - v_now
            adv = self.compute_advantage(self.gamma, self.lmbda, td_del.detach()).to(self.device)

            # 3 – actor & critic losses
            logp   = torch.log(self.actor(states).gather(1, actions))
            ratio  = torch.exp(logp - old_log_probs)
            actor_loss  = -(torch.min(ratio * adv,
                                       torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv)).mean()
            critic_loss = F.mse_loss(v_now, td_tgt)

            # 4 – dynamic-prediction loss  ẑ_t ≈ z_t
            a_prev = F.one_hot(whole_action[:-1], num_classes=predictor.action_dim).float()    # (T-1,N,|A|)
            r_prev = whole_reward[:-1].unsqueeze(-1)                                     # (T-1,N,1)
            z_prev = z_all[:-1].reshape(-1, z_t.size(-1))                             # ((T-1)*N,d)
            z_tgt  = z_all[1:].reshape(-1, z_t.size(-1))
            z_hat  = predictor(z_prev,
                               a_prev.reshape(-1, predictor.action_dim),
                               r_prev.reshape(-1, 1))
            pred_loss = F.mse_loss(z_hat, z_tgt.detach())

            # 5 – back-prop
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            total_loss = actor_loss + critic_loss + self.pred_coef * pred_loss
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item(), pred_loss.item()

    # ---- helpers --------------------------------------------------------
    def get_attn_lr_history(self):       return self.lr_history
    def get_full_attention(self):   return self.full_A

    @staticmethod
    def compute_advantage(gamma, lmbda, td_delta):
        td = td_delta.cpu().numpy()
        adv, out = 0.0, []
        for delta in td[::-1]:
            adv = gamma * lmbda * adv + delta
            out.append(adv)
        adv_ts = torch.tensor(out[::-1], dtype=torch.float32)
        return (adv_ts - adv_ts.mean()) / (adv_ts.std() + 1e-5)
