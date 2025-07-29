import torch, numpy as np, torch.nn.functional as F

class MacLight:
    def __init__(self, policy_net, critic_net,
                 actor_lr=1e-4, critic_lr=5e-3,
                 gamma=0.9, lmbda=0.9,pred_coef=0.3, epochs=20, eps=0.2,
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

    # ───────────────────────────────────────────────────────────────
    # update  —  **new arg accumulate_attn_grad**
    # ───────────────────────────────────────────────────────────────
    def update(self, transition_dict, agent_name,attention,predictor, accumulate_attn_grad=False):
        """One agent’s PPO update.  
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
        stack = lambda key: torch.stack([ to_t(transition_dict[key][a], torch.float32)
                                          for a in agt_order ], dim=1)
        whole_state       = stack('states')
        whole_next_state  = stack('next_states')

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # ------------------------------------------------------------------
        # 3. PPO E epochs
        # ------------------------------------------------------------------
        for epoch in range(self.epochs):

            g_all , A   = attention(whole_state)         # (T,N,d)
            g_next_all,_= attention(whole_next_state)

            g      = g_all      [:, idx, :]                   # (T,d)
            g_next = g_next_all [:, idx, :]

            if epoch == self.epochs-1:                                    # keep last A
                self.full_A = A.detach().cpu().to(torch.float16)  # (T,H,N,N) 
            # Debug 
            


            # critic targets / deltas
            v_now  = self.critic(states, g)
            v_next = self.critic(next_states, g_next).detach()
            td_tgt = rewards + self.gamma * v_next * (1 - dones)
            td_del = td_tgt - v_now
            adv    = self.compute_advantage(self.gamma, self.lmbda,
                                            td_del.detach()).to(self.device)

            # losses
            logp   = torch.log(self.actor(states).gather(1, actions))
            ratio  = torch.exp(logp - old_log_probs)
            actor_loss  = -(torch.min(ratio*adv,
                                      torch.clamp(ratio,1-self.eps,1+self.eps)*adv)).mean()
            critic_loss = F.mse_loss(v_now, td_tgt)

            # ------------- auxiliary predictor loss -------------
            o_pred    = predictor(g,states)                     # (T, obs_dim)
            pred_loss  = F.mse_loss(o_pred, next_states)  # predict next local obs

            # backward
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)

            (actor_loss + critic_loss +self.pred_coef*pred_loss).backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()
            # >>>> Attention step happens **outside** when accumulate=True

        return actor_loss.item(), critic_loss.item(),pred_loss.item()

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
