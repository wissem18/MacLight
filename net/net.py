import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv as GATConv


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

# ---------- NEW: Feature-fusion à la X-Light ------------------------------
class FeatureFusion(nn.Module):
    """Self-attend over 3 token types: obs ‖ prev-action ‖ prev-reward."""
    def __init__(self, d_model=16, d_out=32):
        super().__init__()
        self.q = nn.Parameter(torch.randn(3, d_model))      # (3,d)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_out)

    def forward(self, tokens):                # (B,N,3,d_model)
        Q = self.q                            # (3,d)
        K = self.k(tokens)                    # (B,N,3,d)
        V = self.v(tokens)                    # (B,N,3,d)
        attn = (Q @ K.transpose(-1, -2)) / (tokens.size(-1) ** 0.5)   # (3,3)
        w = attn.softmax(-1)                                          # (3,3)
        fused = (w @ V).mean(-2)               # (B,N,d_model)
        return self.proj(F.relu(fused))        # (B,N,d_out)

# ------------------------------------------------------------------
# Observation → Embedding   (2 × 32-unit MLP + ReLU)
# ------------------------------------------------------------------
class ObsEmbedding(nn.Module):
    """
    X-Light style:   [ local-obs(33) | prev-action(one-hot) | prev-reward ]
    Each slice → 16-d, self-attn fuse → post-proj (Linear+ReLU+LN) → 32.
    """
    def __init__(self, d_obs=33, a_dim=8, d_model=16, d_out=32):
        super().__init__()
        self.a_dim = a_dim
        self.fc_s = nn.Linear(d_obs,  d_model)
        self.fc_a = nn.Linear(a_dim,  d_model)
        self.fc_r = nn.Linear(1,      d_model)
        self.fuse = FeatureFusion(d_model, d_out)

        # post-projection
        self.post = nn.Sequential(
            nn.Linear(d_out, d_out),
            nn.ReLU(),
            nn.LayerNorm(d_out)
        )

    def forward(self, x):                       # x : (B,N, 33 + a_dim + 1)
        s, a, r = torch.split(x, [33, self.a_dim, 1], dim=-1)
        tokens = torch.stack([self.fc_s(s), self.fc_a(a), self.fc_r(r)], dim=-2)
        z = self.fuse(tokens)                  # (B,N,32)
        return self.post(z)                    # (B,N,32)   <-- final
    

class GATBlock(nn.Module):
    """
    Sparse multi-head Graph Attention.
    """
    def __init__(self, d_out=32, heads=4, a_dim=8,
                 edge_index=None, dropout=0.1):
        super().__init__()
        assert edge_index is not None, "edge_index required"
        self.register_buffer("edge_index", edge_index)      # (2,E), no grad
        # --- X-Light style embedding we defined earlier ----------
        self.embed = ObsEmbedding(d_obs=33, a_dim=a_dim,    # d_model=16 by default
                                  d_model=16, d_out=32)

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