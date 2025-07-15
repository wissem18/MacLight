import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


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
    

# ------------------------------------------------------------------
# Two-hop GATv2 block  (embed → GAT → ReLU → GAT → +residual → LN)
# ------------------------------------------------------------------
class TwoHopGATBlock(nn.Module):
    def __init__(self, d_in=33, d_mid=32, d_out=32, heads=4,
                 edge_index=None, dropout=0.1):
        super().__init__()
        assert edge_index is not None, "edge_index required"
        self.register_buffer("edge_index", edge_index)      # (2,E)
        self.embed = ObsEmbedding(d_in, 32)                 # 32-d embed

        # 1-hop
        self.gat1 = GATConv(32, d_mid // heads, heads=heads,
                            dropout=dropout, add_self_loops=False)

        # 2-hop
        self.gat2 = GATConv(d_mid, d_out // heads, heads=heads,
                            dropout=dropout, add_self_loops=False)

        self.norm = nn.LayerNorm(d_out)                     # residual-LN
        self.heads = heads

    def forward(self, H):           # H : (B,N,d_in)
        B, N, _ = H.size()
        H = self.embed(H)           # (B,N,32)

        outs, attn = [], []
        for b in range(B):
            # ----- first layer (1-hop) ----------------------------
            h1, _ = self.gat1(H[b], self.edge_index,
                              return_attention_weights=True)
            h1 = F.relu(h1)

            # ----- second layer (2-hop) ---------------------------
            h2, (e_idx, α_b) = self.gat2(
                    h1, self.edge_index, return_attention_weights=True)

            # ----- residual + LayerNorm ---------------------------
            h_out = self.norm(h1 + h2)          # (N,d_out)

            # build dense attention (heads,H,N,N) from 2-hop α_b
            A_dense = torch.zeros(self.heads, N, N, device=H.device)
            for h in range(self.heads):
                A_dense[h][e_idx[0], e_idx[1]] = α_b[:, h]

            outs.append(h_out)
            attn.append(A_dense)

        return torch.stack(outs, 0), torch.stack(attn, 0)

    
    
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