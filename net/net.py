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


# VAE 模型
class VAE(nn.Module):
    def __init__(self, state_dim=33, latent_dim=16):
        super(VAE, self).__init__()

        # 编码器部分：Convolutional Layers
        # 输入: [batch_size, 33, 4, 4]
        self.encoder = nn.Sequential(
            nn.Conv2d(state_dim, 64, kernel_size=3, stride=1, padding=1),  # [batch_size, 64, 4, 4]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [batch_size, 128, 4, 4]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # [batch_size, 256, 2, 2]
            nn.ReLU()
        )

        # Flatten卷积输出并映射到潜在空间
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)        # 均值
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)    # 对数方差

        # 将潜在向量映射回到解码器的输入维度
        self.fc_decode = nn.Linear(latent_dim, 256 * 2 * 2)

        # 解码器部分：Transposed Convolutional Layers（反卷积）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch_size, 128, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # [batch_size, 64, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 33, kernel_size=3, stride=1, padding=1),  # [batch_size, 33, 4, 4]
            nn.Sigmoid()  # 输出范围在 [0, 1]，适用于归一化的图像数据
        )

    def encode(self, x):
        """编码器：提取输入并生成潜在变量的均值和方差"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # 展平成一维向量
        mu = self.fc_mu(x)         # 均值
        logvar = self.fc_logvar(x) # 对数方差
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """通过 reparameterization trick 采样潜在变量"""
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)    # 从标准正态分布中采样
        return mu + eps * std          # 采样潜在向量

    def decode(self, z):
        """解码器：将潜在变量映射回原始输入的维度"""
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 2, 2)  # reshape回卷积层输入维度
        x = self.decoder(x)
        return x

    def forward(self, x):
        """前向传播：编码 -> 采样 -> 解码"""
        mu, logvar = self.encode(x)        # 编码阶段
        z = self.reparameterize(mu, logvar)  # 采样阶段
        recon_x = self.decode(z)           # 解码阶段
        return recon_x, mu, logvar

    def representation(self, x):
        return self.encode(x)[0].detach()


class VAnet(torch.nn.Module):
    '''For IDQN'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(-1).view(-1, 1)  # Q值由V值和A值计算得到
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