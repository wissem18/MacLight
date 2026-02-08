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
        self.full_A =None



    # ───────────────────────────────────────────────────────────────
    # take_action unchanged
    # ───────────────────────────────────────────────────────────────
    def take_action(self, state):
        state  = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.distributions.Categorical(self.actor(state)).sample()
        return action.item()

    
    def update(self, transition_dict, agent_name, attention, temp_enc, predictor, accumulate_attn_grad=False):
        """
        One agent’s PPO update.
        NOW USES: Fixed Beta Reward (25% Ego + 75% Mean Neighbor).
        """
        # 1. Setup Data
        to_t = lambda x, dtype: torch.tensor(x, dtype=dtype, device=self.device)
        states      = to_t(transition_dict['states'][agent_name]     , torch.float32)
        actions     = to_t(transition_dict['actions'][agent_name]    , torch.int64 ).view(-1,1)
        # Load raw rewards (will be overwritten)
        rewards     = to_t(transition_dict['rewards'][agent_name]    , torch.float32).view(-1,1)
        next_states = to_t(transition_dict['next_states'][agent_name], torch.float32)
        dones       = to_t(transition_dict['dones'][agent_name]      , torch.int32 ).view(-1,1)

        # 2. Build Whole-Network Data (for Neighbor Calculation)
        # Use simple list order (relies on env.agents order consistency)
        agt_order = list(transition_dict['states'].keys())
        idx       = agt_order.index(agent_name)
        
        stack = lambda key, dtype=torch.float32: torch.stack([to_t(transition_dict[key][a], dtype) for a in agt_order], dim=1)
        
        whole_state      = stack('states')
        whole_next_state = stack('next_states')
        whole_action     = stack('actions', dtype=torch.int64)
        whole_reward_raw = stack('rewards')  # Shape: (T, N) or (T, N, 1) depending on view

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # ==================================================================
        # [NEW] FIXED REWARD CALCULATION (Beta Strategy)
        # Formula: R_new = beta * R_ego + (1-beta) * Mean(R_neighbors)
        # ==================================================================
        
        # A. Define Beta
        beta = 0.25  # 25% Ego, 75% Neighbors
        
        # B. Build Static Adjacency Matrix from GAT's edge_index
        # attention.edge_index is (2, E)
        N_nodes = whole_reward_raw.shape[1]
        adj = torch.zeros((N_nodes, N_nodes), device=self.device)
        
        # Fill 1.0 where edges exist (Binary Adjacency)
        # This assumes edge_index contains directed edges i->j
        adj.index_put_((attention.edge_index[0], attention.edge_index[1]), 
                       torch.tensor(1.0, device=self.device))
        
        # Remove self-loops from adjacency if they exist (to strictly calculate neighbor mean)
        adj.fill_diagonal_(0)

        # C. Calculate Mean Neighbor Reward
        # Sum of neighbor rewards for each node: Matrix Mul (N,N) x (T,N,1) -> (T,N,1)
        # We ensure whole_reward_raw is (T, N, 1)
        if whole_reward_raw.dim() == 2:
            whole_reward_raw = whole_reward_raw.unsqueeze(-1)
            
        sum_neighbor_R = torch.matmul(adj, whole_reward_raw) 
        
        # Calculate Degree (count of neighbors) for each node
        degree = adj.sum(dim=1, keepdim=True) # Shape (N, 1)
        degree = torch.clamp(degree, min=1.0) # Avoid division by zero
        
        # Mean = Sum / Degree
        # Transpose degree to broadcast correctly against (T, N, 1) if needed, 
        # but usually (N,1) broadcasts fine against (T,N,1) on the 2nd dim.
        mean_neighbor_R = sum_neighbor_R / degree

        # D. Apply Formula
        R_fixed_all = (beta * whole_reward_raw) + ((1 - beta) * mean_neighbor_R)

        # E. Extract THIS agent's new reward
        rewards = R_fixed_all[:, idx].view(-1, 1)
        
        # [Optional Debug]
        # if idx == 0:
        #     print(f"Agent {idx} Raw: {whole_reward_raw[0,idx].item():.2f} | NeighMean: {mean_neighbor_R[0,idx].item():.2f} | Final: {rewards[0].item():.2f}")

        # ==================================================================

        # 3. PPO Epochs
        for epoch in range(self.epochs):
            # 1– Spatial embeddings via GAT 
            # (We still run GAT for the State Embedding, even if we don't use it for Reward)
            h_all, _ = attention(whole_state) 

            # Stack K for Temporal Encoder
            K = temp_enc.K
            pad  = torch.zeros(K - 1, *h_all.shape[1:], device=self.device)
            cat  = torch.cat([pad, h_all], dim=0)
            hist = torch.stack([cat[t:t + K] for t in range(h_all.size(0))], 0)
            hist = hist.permute(0, 2, 1, 3)
            z_all = temp_enc(hist)

            z_t     = z_all[:, idx, :]
            z_next  = torch.cat([z_t[1:], z_t[-1:].clone()], dim=0)

            # 2– Critic targets (Uses Fixed Rewards)
            v_now  = self.critic(states, z_t)
            v_next = self.critic(next_states, z_next).detach()
            td_tgt = rewards + self.gamma * v_next * (1 - dones)
            td_del = td_tgt - v_now
            adv = self.compute_advantage(self.gamma, self.lmbda, td_del.detach()).to(self.device)

            # 3– Actor & Critic losses
            logp   = torch.log(self.actor(states).gather(1, actions))
            ratio  = torch.exp(logp - old_log_probs)
            actor_loss  = -(torch.min(ratio * adv,
                                     torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv)).mean()
            critic_loss = F.mse_loss(v_now, td_tgt)

            # 4– Dynamic-prediction loss
            a_prev = F.one_hot(whole_action[:-1], num_classes=predictor.action_dim).float()
            r_prev = whole_reward_raw[:-1] # Use RAW rewards for predictor physics
            z_prev = z_all[:-1].reshape(-1, z_t.size(-1))
            z_tgt  = z_all[1:].reshape(-1, z_t.size(-1))
            z_hat  = predictor(z_prev,
                               a_prev.reshape(-1, predictor.action_dim),
                               r_prev.reshape(-1, 1))
            pred_loss = F.mse_loss(z_hat, z_tgt.detach())

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
