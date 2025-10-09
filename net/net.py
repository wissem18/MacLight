import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv as GATConv
from torch_geometric.data import Data, Batch
import sys
sys.path.append("external/SAT")  
from sat import GraphTransformer
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


class SATBlock(nn.Module):
    """
    Structure-Aware Transformer spatial encoder.
      forward(H) -> (B, N, d_out)
    Notes:
      - Uses the official GraphTransformer (node-level outputs, no classifier head).
      - No edge features required.
    """
    def __init__(self, d_in=33, d_out=32,
                 edge_index=None, dropout=0.1,
                 k_hop=2, gnn_type="pna", num_layers=2):
        super().__init__()
        assert edge_index is not None, "edge_index required"
        self.register_buffer("edge_index", edge_index)     # (2, E), long, no grad
        self.embed = ObsEmbedding(d_in=d_in, hidden_size=32)

        # IMPORTANT: We want node-level embeddings → num_class=None and no global_pool in GraphTransformer
        self.sat = GraphTransformer(
            in_size=32,                  # we feed the 32-dim ObsEmbedding output
            num_class=d_out,              # no classifier head
            d_model=d_out,               # final per-node embedding dim == d_out
            dim_feedforward=2 * d_out,
            num_layers=num_layers,       # SAT transformer depth
            batch_norm=True,
            gnn_type=gnn_type,           # "pna" | "gcn" | "gine"
            use_edge_attr=False,         # you said no edge features
            num_edge_features=0,
            edge_dim=d_out,              # irrelevant when use_edge_attr=False
            k_hop=k_hop,                 # structure extractor depth
            se="gnn",                # or "khopgnn"; start with "gnn"
            dropout=dropout,
            global_pool=None       # ← IMPORTANT: disable graph pooling to keep node outputs
        )
        class IdentityPool(nn.Module):
            def forward(self, x, batch):
                return x
        self.sat.embedding = torch.nn.Identity()
        # 2) Disable graph pooling but keep the call site happy
        self.sat.pooling = IdentityPool()

    def forward(self, H):                 # H: (B, N, d_in)
        B, N, _ = H.size()
        H_emb = self.embed(H)             # (B, N, 32)
        # Run SAT per batch item (shared graph, different features)
        outs = []
        for b in range(B):
            data = Data(x=H_emb[b], edge_index=self.edge_index)  # x: (N,32)
             # IMPORTANT: make a Batch so .ptr exists
            batch = Batch.from_data_list([data])
            batch = batch.to(H.device)
            z_b = self.sat(batch)                                 # (N, d_out)
            outs.append(z_b)
        Z = torch.stack(outs, dim=0)                             # (B, N, d_out)
        # For API parity with your GATBlock, return attn=None
        return Z,None


class GATBlock(nn.Module):
    """
    Sparse multi-head Graph Attention.
    """
    def __init__(self, d_in=33, d_out=32, heads=4,
                 edge_index=None, dropout=0.1):
        super().__init__()
        assert edge_index is not None, "edge_index required"
        self.register_buffer("edge_index", edge_index)      # (2,E), no grad
        self.embed = ObsEmbedding(d_in=d_in,hidden_size=32)

        # Each head outputs d_out / heads features, concatenated → d_out
        self.gat = GATConv(
            in_channels=32,
            out_channels=d_out // heads,
            heads=heads,
            dropout=dropout,
            add_self_loops=False
        )

    def forward(self, H):               # H: (B, N, d_in)
        B, N, _ = H.size()
        H = self.embed(H)
        out = []
        attn = []
        h_=self.gat.heads
        for b in range(B):
            # out_b : (N,d_out)   α_b : (E,heads)
            out_b, (e_idx, α_b) = self.gat(
                    H[b], self.edge_index, return_attention_weights=True)

            # α_b : (E, H)  →   build dense (H, N, N) where H is the number of heads
            A_dense = torch.zeros(h_, N, N, device=H.device)
            
            for i in range(h_):
                A_dense[i].index_put_((e_idx[0], e_idx[1]),
                                  α_b[:, i],
                                  accumulate=True)
            # renormalise rows
            A_dense = A_dense / A_dense.sum(-1, keepdim=True).clamp(min=1e-9)

            out.append(out_b)
            attn.append(A_dense)             

        return torch.stack(out, 0), torch.stack(attn, 0)   # (B,N,d) , (B,H,N,N)
    
# ─────────────────────────────────────────────────────────────
#  Temporal Encoder  (Upper Transformer from X-Light)
# ─────────────────────────────────────────────────────────────
class TemporalEncoder(nn.Module):
    """
    Attend over the last K GAT outputs per node.
      • input : (B, N, K, d)   — stack from a deque
      • output: (B, N, d)      — last-token after attention
    """
    def __init__(self, d_model: int = 32, K: int = 8,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.pos = nn.Parameter(torch.zeros(K, d_model))     # learnable PE
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, activation="relu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        # hist : (B, N, K, d)
        B, N, K, d = hist.shape
        x = (hist + self.pos).contiguous().view(B * N, K, d)
        z = self.encoder(x)[:, -1]                            # last token
        return z.view(B, N, d)


# ─────────────────────────────────────────────────────────────
#  Dynamic Predictor  (z_{t-1}, a_{t-1}, r_{t-1} → ẑ_t)
# ─────────────────────────────────────────────────────────────
class DynamicPredictor(nn.Module):
    """
    Predict next transformer output  z_t  from
    [ z_{t-1} ‖ onehot(a_{t-1}) ‖ r_{t-1} ].
      • input dims : d_model + |A| + 1
      • output     : d_model
    """
    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        hidden = 4 * d_model
        self.action_dim=action_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_model + action_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, z_prev: torch.Tensor,
                a_prev_oh: torch.Tensor,
                r_prev: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_prev   : (B*N, d_model)
        a_prev_oh: (B*N, |A|)
        r_prev   : (B*N, 1)
        """
        inp = torch.cat([z_prev, a_prev_oh, r_prev], dim=-1)
        return self.mlp(inp)
    
    
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