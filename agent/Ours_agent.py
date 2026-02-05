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

    
    def update(self, transition_dict, agent_name, attention, temp_enc, predictor, accumulate_attn_grad=False):
        """
        One agent’s PPO update with temporal encoder + dynamic predictor.
        INCLUDES: Attention-Weighted Reward Calculation.
        """
        # ------------------------------------------------------------------
        # 1. Pull this agent’s local trajectory tensors (RAW REWARDS)
        # ------------------------------------------------------------------
        to_t = lambda x, dtype: torch.tensor(x, dtype=dtype, device=self.device)
        states      = to_t(transition_dict['states'][agent_name]     , torch.float32)
        actions     = to_t(transition_dict['actions'][agent_name]    , torch.int64 ).view(-1,1)
        # Load local rewards (will be overwritten later with enhanced rewards)
        rewards     = to_t(transition_dict['rewards'][agent_name]    , torch.float32).view(-1,1)
        next_states = to_t(transition_dict['next_states'][agent_name], torch.float32)
        dones       = to_t(transition_dict['dones'][agent_name]      , torch.int32 ).view(-1,1)

        # ------------------------------------------------------------------
        # 2. Build whole-state tensors (T, N, ...) for shared Attention
        # ------------------------------------------------------------------
        # [USER REQUEST] No sorting, relying on preservation of insertion order
        agt_order = list(transition_dict['states'].keys())
        idx       = agt_order.index(agent_name)
        
        stack = lambda key, dtype=torch.float32: torch.stack([to_t(transition_dict[key][a], dtype) for a in agt_order], dim=1)
        
        whole_state      = stack('states')
        whole_next_state = stack('next_states')
        whole_action     = stack('actions', dtype=torch.int64)
        
        # Stack ALL agents' raw rewards to compute neighbors' contribution
        whole_reward_raw = stack('rewards')  # Shape: (T, N)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # ------------------------------------------------------------------
        # 3. PPO E epochs
        # ------------------------------------------------------------------
        for epoch in range(self.epochs):

            # 1– Spatial embeddings via GAT
            # h_all: (T, N, d), A: (T, Heads, N, N)
            h_all, A = attention(whole_state) 

            # ==================================================================
            # [NEW] ATTENTION-WEIGHTED REWARD CALCULATION
            # Formula: R_new = R_self + sum(alpha_iv * R_v)
            # ==================================================================
            
            # Average attention heads to get (T, N, N)
            alpha_matrix = A.mean(dim=1).detach() 
            
            # Calculate Neighbor Contribution: (T,N,N) x (T,N,1) -> (T,N,1)
            R_neighbors = torch.bmm(alpha_matrix, whole_reward_raw.unsqueeze(-1)).squeeze(-1)
            
            # Add Raw Reward + Neighbor Contribution
            R_enhanced_all = whole_reward_raw + R_neighbors
            
            # Extract ONLY the current agent's new reward for this update
            rewards_enhanced = R_enhanced_all[:, idx].view(-1, 1)

            # --- [DEBUG BLOCK: RUNS ONCE PER UPDATE CALL] ---
            if epoch == 9:
                print(f"\n--- [DEBUG] Agent: {agent_name} (Index {idx}) ---")
                
                # Check 1: Neighbors with non-zero attention (at t=0)
                # alpha_matrix[0, idx] is the row of attention weights THIS agent gives to others
                my_attn = alpha_matrix[0, idx]
                neighbor_indices = torch.nonzero(my_attn > 0.001).flatten()
                print(f"  > Connected Neighbors (Indices): {neighbor_indices.tolist()}")
                print(f"  > Attention Weights: {my_attn[neighbor_indices].cpu().numpy().round(3)}")
                
                # Check 2: Reward Values (at t=0)
                raw_val = whole_reward_raw[0, idx].item()
                neighbor_contrib_val = R_neighbors[0, idx].item()
                final_val = R_enhanced_all[0, idx].item()
                
                print(f"  > Raw Reward (R_it): {raw_val:.4f}")
                print(f"  > Neighbor Contrib (sum(alpha*R_vt)): {neighbor_contrib_val:.4f}")
                print(f"  > Final Reward (R_enhanced): {final_val:.4f}")
                print("--------------------------------------------------\n")
            # --------------------------------------------------------------

            # Overwrite the 'rewards' variable used for TD Target calculation
            rewards = rewards_enhanced
            # ==================================================================

            # Stack the last K GAT outputs for each timestep -> Transformer
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

            # 2– Critic targets (Uses the ENHANCED 'rewards')
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
            # NOTE: We use RAW rewards for prediction task to keep physics consistent
            a_prev = F.one_hot(whole_action[:-1], num_classes=predictor.action_dim).float()
            r_prev = whole_reward_raw[:-1].unsqueeze(-1)  # Using RAW rewards for predictor
            z_prev = z_all[:-1].reshape(-1, z_t.size(-1))
            z_tgt  = z_all[1:].reshape(-1, z_t.size(-1))
            z_hat  = predictor(z_prev,
                               a_prev.reshape(-1, predictor.action_dim),
                               r_prev.reshape(-1, 1))
            pred_loss = F.mse_loss(z_hat, z_tgt.detach())

            # 5– Back-prop
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
