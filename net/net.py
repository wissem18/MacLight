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


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, global_emb_dim=0):
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
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),           # second hidden layer
            nn.ReLU(inplace=True)
        )
        # output dim = 32
    def forward(self, x):                # x : (B,N,d_in)
        return self.mlp(x)               # (B,N,32)

# ------------------------------------------------------------------
# Single-head (fully-connected) neighbour Attention
# ------------------------------------------------------------------
#  • H contains *local+positional* tokens for ALL 16 intersections
#    of the grid at the current time-step.
#  • For each intersection i we want a **global embedding** g_i that is
#    a weighted combination of the OTHER intersections’ features.
#  • We forbid self-attention (i → i) by masking the diagonal.
#  • Output dimension d_out is configurable (here 32).  This is the
#    vector you will later concatenate with the 33-dim local state
#    before feeding the critic / actor.
#
#  Tunable hyper-params
#  --------------------
#  d_in   : dim of each input token (33 local + 8 pos = 41 in our setup)
#  d_a    : hidden/“attention” size used for Q,K,V projections
#           (64 is a standard small value;)
#  d_out  : size of final global vector g_i  (32)
#
# ------------------------------------------------------------------
class Attention(nn.Module):
    """
    Parameters
    ----------
    d_in  : int   dimension of input token  (default 41)
    d_a   : int   dimension of Q / K / V projections (default 64)
    d_out : int   dimension of aggregated neighbour vector (default 32)

    Forward
    -------
    H : Tensor, shape (B, N, d_in)
        B = batch ; N = number of agents (16 here)

    Returns
    -------
    G : Tensor, shape (B, N, d_out)
        weighted neighbour vector for every agent
    A : Tensor, shape (B, N, N)
        softmax attention weights (for analysis)
    """
    def __init__(self, d_in: int = 41, d_a: int = 64, d_out: int = 32):
        super().__init__()
        # ----shared embedding ----
        self.embed = ObsEmbedding(d_in, 32)   # 32-dim output
        # Linear projections for Query, Key, Value
        self.W_q = nn.Linear(32, d_a, bias=False)
        self.W_k = nn.Linear(32, d_a, bias=False)
        self.W_v = nn.Linear(32, d_a, bias=False)
        # Post-aggregation transform + ReLU (adds non-linearity, lets you
        # pick any output size d_out ≠ d_a)
        self.W_o = nn.Linear(d_a, d_out, bias=True)
        self.scale = 1.0 / (d_a ** 0.5)        # √d_a : Transformer norm

    def forward(self, H: torch.Tensor):
        """
        H expected shape: [B, N, d_in]
        """
        self.debug_H=H.detach()
        H = self.embed(H)                     # (B,N,32)
        self.debug_embedded_H=H.detach()
        # 1. Q, K, V projections
        Q = self.W_q(H)        # [B, N, d_a]
        K = self.W_k(H)        # [B, N, d_a]
        V = self.W_v(H)        # [B, N, d_a]

        # 2. Scaled dot-product attention scores
        #    scores[b,i,j] = (q_i · k_j) / √d_a
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,N,N]
        self.debug_scores=scores.clone().detach()
        # 3. Mask the diagonal so each agent ignores itself
        diag = torch.eye(scores.size(-1), dtype=torch.bool, device=scores.device)
        scores.masked_fill_(diag.unsqueeze(0), float('-inf'))

        # 4. Soft-max → weights
        A = torch.softmax(scores, dim=-1)      # [B, N, N]

        # 5. Weighted sum of neighbour values  ->  g_i
        G = torch.matmul(A, V)                 # [B, N, d_a]

        # 6. Final linear + ReLU 
        G = torch.relu(self.W_o(G))            # [B, N, d_out]

        return G, A.detach()   # return attention weights for analysis


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