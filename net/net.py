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

# Attention Mechanism
class Attention(nn.module):
    def __init__(self, feature_dim=33, num_agents=16):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_agents = num_agents
        
        # Linear transformations for query, key, and value
        self.W_q = nn.Linear(feature_dim, feature_dim)  # Query transformation 
        self.W_k = nn.Linear(feature_dim, feature_dim)  # Key transformation 
        self.W_v = nn.Linear(feature_dim, feature_dim)  # Value transformation 
        
        # Scaling factor for attention scores
        self.tau = feature_dim ** 0.5  # Scaled by sqrt(feature_dim) for stability

    def forward(self, features):
        """
        Compute attention-based context vectors where the query comes from agent i,
        and keys and values come from other agents j ≠ i.
        
        Args:
            features: Tensor of shape (batch_size, num_agents, feature_dim)
        
        Returns:
            context: Tensor of shape (batch_size, num_agents, feature_dim)
                     Attention-applied output for each agent
        """
        batch_size, num_agents, feature_dim = features.size()
        
        # Step 1: Compute query for each agent i
        # Shape: (batch_size, num_agents, feature_dim)
        q = self.W_q(features)
        
        # Step 2: Prepare indices to exclude self (agent i) for keys and values
        # We will compute keys and values for all agents first, then mask out i
        k = self.W_k(features)  # Shape: (batch_size, num_agents, feature_dim)
        v = self.W_v(features)  # Shape: (batch_size, num_agents, feature_dim)
        
        # Step 3: Compute attention scores between q_i and k_j for all j
        # Shape: (batch_size, num_agents, num_agents)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.tau
        
        # Step 4: Mask out self-attention (where i == j)
        # Create a diagonal mask: True where i == j
        mask = torch.eye(num_agents, device=features.device).bool()
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, num_agents, num_agents)
        
        # Set scores where i == j to a large negative value to exclude self
        scores = scores.masked_fill(mask, -1e9)
        
        # Step 5: Compute attention weights using softmax
        # Shape: (batch_size, num_agents, num_agents)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 6: Compute context vectors using values from other agents
        # Multiply attention weights with values v_j
        # Shape: (batch_size, num_agents, feature_dim)
        context = torch.matmul(attention_weights, v)
        
        # Step 7: Apply ReLU activation to the context vectors
        context = F.relu(context)
        
        return context    

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