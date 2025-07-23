from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv as GATConv
from torch_scatter import topk as scatter_topk

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.h_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.h_1(F.relu(self.fc1(x))))
        return F.softmax(self.fc2(x), dim=-1)

# Add attention block for the ValueNet
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim,global_emb_dim=0):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim+global_emb_dim, hidden_dim)
        self.h_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, global_emb=None):
        x = torch.cat([x, global_emb], dim=1) if global_emb is not None  else x
        x = F.relu(self.h_1(F.relu(self.fc1(x))))
        return self.fc2(x)

# ------------------------------------------------------------------
# Observation → Embedding   (2 × 32-unit MLP + ReLU)
# ------------------------------------------------------------------
class ObsEmbedding(nn.Module):
    """
    Embeds the local observation
    Hidden size = 32 
    """
    def __init__(self, d_in: int, hidden_size: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),           # second hidden layer
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        # output dim = 32
    def forward(self, x):                # x : (B,N,d_in)
        return self.mlp(x)               # (B,N,32)
    

# -----------------------------------------------------------------------------#
# 2.  PER-NODE 12-STEP GRU MEMORY (shared weights, N hidden vectors)            #
# -----------------------------------------------------------------------------#
class NodeMemory(nn.Module):
    """
    One GRUCell whose weights are shared by all agents.
    Hidden state tensor h ∈ ℝ^{N×H} is stored inside the module.
    Call .reset(N, device) at the beginning of every episode.
    """
    def __init__(self, in_dim: int, hid_dim: int = 64, tbptt: int = 12):
        super().__init__()
        self.gru   = nn.GRUCell(in_dim, hid_dim)
        self.h     = None            # (N,H) – filled by reset()
        self.t     = 0
        self.tbptt = tbptt

    def reset(self, num_nodes: int, device):
        self.h = torch.zeros(num_nodes, self.gru.hidden_size, device=device)
        self.t = 0

    def forward(self, e_t: torch.Tensor):     # e_t : (N,in_dim)
        if (self.t % self.tbptt) == 0:
            self.h = self.h.detach()          # truncate BPTT every 12 steps
        self.h = self.gru(e_t, self.h)
        self.t += 1
        return self.h                         # (N,hid_dim)
    
# ------------------------------------------------------------
# 3. LEARNABLE EDGE SCORER  (MAGSAC-style)
# ------------------------------------------------------------
class EdgeScorer(nn.Module):
    """
    edge_candidates : tuple(src,dst) – indices of ≤2-hop pairs (torch tensors)
    k               : keep at most k neighbours per source node every step
    """
    def __init__(self,
                 h_dim: int,
                 edge_candidates: Tuple[torch.Tensor, torch.Tensor],
                 k: int = 4):
        super().__init__()
        self.register_buffer('src', edge_candidates[0])
        self.register_buffer('dst', edge_candidates[1])
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2 * h_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, h: torch.Tensor):
        """
        h : (N, h_dim) – node hidden states
        Returns
        -------
        edge_index : (2,E_kept)
        edge_w     : (E_kept,)   – soft weights in [0,1]
        """
        feat   = torch.cat([h[self.src], h[self.dst]], dim=1)   # (E,2h)
        score  = torch.sigmoid(self.mlp(feat).squeeze())        # (E,)

        keep   = scatter_topk(score, self.k, index=self.src)    # indices kept
        edge_i = torch.stack([self.src[keep], self.dst[keep]], 0)
        return edge_i, score[keep]
        
# ------------------------------------------------------------
# 4. DYNAMIC GNN LAYER  (Embed ➜ Node-GRU ➜ EdgeScorer ➜ GATv2)
# ------------------------------------------------------------
class DynamicGNN(nn.Module):
    """
    forward(H)  where H is (B,N,d_in)

    returns:
        out   : (B,N,hid_dim)
        attn  : (B, heads, N, N)   – dense softmaxed attention
    """
    def __init__(self,
                 obs_dim: int = 33,
                 hid_dim: int = 64,
                 out_dim: int = 32,
                 heads: int = 4,
                 edge_candidates: Tuple[torch.Tensor, torch.Tensor] = None,
                 k: int = 4,
                 dropout: float = 0.1,
                 device: str = 'cpu'):
        super().__init__()
        assert edge_candidates is not None, "edge_candidates required"

        self.embed   = ObsEmbedding(obs_dim, 32)
        self.memory  = NodeMemory(32, hid_dim)
        self.scorer  = EdgeScorer(hid_dim, edge_candidates, k)
        self.gat     = GATConv(in_channels=hid_dim,
                                 out_channels=out_dim // heads,
                                 heads=heads,
                                 dropout=dropout,
                                 add_self_loops=False)
        self.device  = device
        # allocate hidden state (will be zeroed by reset())
        self.memory.reset(num_nodes=len(edge_candidates[0].unique()), device=device)

    def reset_memory(self):
        self.memory.reset(num_nodes=self.memory.h.size(0), device=self.device)

    # --------------------------------------------------------------------- #
    def _dense_attn(self, e_idx, alpha, N, H):
        """
        Build (H,N,N) dense attention from sparse α (E,H).
        """
        A = torch.zeros(H, N, N, device=alpha.device)
        for h in range(H):
            A[h].index_put_((e_idx[0], e_idx[1]), alpha[:, h], accumulate=True)
        A = A / A.sum(-1, keepdim=True).clamp(min=1e-9)
        return A
    # --------------------------------------------------------------------- #

    def forward(self, H_t: torch.Tensor):
        """
        H_t : (B, N, obs_dim)
        """
        B, N, _ = H_t.size()
        H_emb   = self.embed(H_t)          # (B,N,32)
        outs, attns = [], []

        for b in range(B):
            # 1. update per-node memory (shared across batch dims)
            h_nodes = self.memory(H_emb[b])            # (N,hid_dim)

            # 2. dynamic edge set for this step
            edge_idx, w = self.scorer(h_nodes)         # (2,E), (E,)

            # 3. GATv2 (out_b : (N,hid_dim_out))
            out_b, (ei, α_b) = self.gat(
                h_nodes, edge_idx, edge_weight=w,
                return_attention_weights=True)

            # 4. dense α   →  (heads,N,N)
            attn_dense = self._dense_attn(ei, α_b, N, self.gat.heads)

            outs.append(out_b)
            attns.append(attn_dense)

        return torch.stack(outs, 0), torch.stack(attns, 0)   # (B,N,dim) (B,H,N,N)
    
# VAE
class VAE(nn.Module):
    def __init__(self, state_dim=33, latent_dim=16):
        super(VAE, self).__init__()

        # Encoder: Convolutional Layers
        # Input: [batch_size, 33, 4, 4]
        self.encoder = nn.Sequential(
            nn.Conv2d(state_dim, 64, kernel_size=3, stride=1, padding=1),  # [batch_size, 64, 4, 4]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [batch_size, 128, 4, 4]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # [batch_size, 256, 2, 2]
            nn.ReLU()
        )

        # Flatten the convolution output and map it to the latent space
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)        # Mean
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)    # std

        # Map the latent vector back to the decoder input dimension
        self.fc_decode = nn.Linear(latent_dim, 256 * 2 * 2)

        # Decoder: Transposed Convolutional Layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch_size, 128, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # [batch_size, 64, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 33, kernel_size=3, stride=1, padding=1),  # [batch_size, 33, 4, 4]
            nn.Sigmoid()  # The output range is [0, 1], as same as input feature range
        )

    def encode(self, x):
        """Encoder: takes input and generates the mean and variance of the latent variable"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten into a one-dimensional vector
        mu = self.fc_mu(x)         # mean
        logvar = self.fc_logvar(x) # log_std
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sampling latent variables via the reparameterization trick"""
        std = torch.exp(0.5 * logvar)  # Calculate the standard deviation
        eps = torch.randn_like(std)    # Sampling from a standard normal distribution
        return mu + eps * std          # Sample latent vectors

    def decode(self, z):
        """Decoder: Maps latent variables back to the dimensions of the original input"""
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 2, 2)  # Reshape back the convolution layer input dimension
        x = self.decoder(x)
        return x

    def forward(self, x):
        """Forward propagation: Encoding -> Sampling -> Decoding"""
        mu, logvar = self.encode(x)        # Encode
        z = self.reparameterize(mu, logvar)  # Sampling
        recon_x = self.decode(z)           # Decode
        return recon_x, mu, logvar

    def representation(self, x):
        return self.encode(x)[0].detach()


class VAnet(torch.nn.Module):
    '''For IDQN'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # sharing part
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(-1).view(-1, 1)  # The Q value is calculated from the V and A values
        return Q


class Qnet(torch.nn.Module):
    def __init__(self, state_dim=4, hidden_dim=128, action_dim=2):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x