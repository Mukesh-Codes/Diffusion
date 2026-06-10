import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t_emb):
        h = self.relu(self.conv1(x))
        time_emb = self.relu(self.time_mlp(t_emb)).unsqueeze(-1)
        h = h + time_emb
        return self.relu(self.conv2(h))

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.down1 = ConvBlock(in_channels, hidden_dim, hidden_dim)
        self.pool1 = nn.MaxPool1d(2)
        self.down2 = ConvBlock(hidden_dim, hidden_dim * 2, hidden_dim)
        self.pool2 = nn.MaxPool1d(2)
        self.bottleneck = ConvBlock(hidden_dim * 2, hidden_dim * 2, hidden_dim)
        self.up1 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(hidden_dim * 2, hidden_dim, hidden_dim)
        self.up2 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(hidden_dim + in_channels, hidden_dim, hidden_dim)
        self.final = nn.Conv1d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, x, k):
        t_emb = self.time_mlp(k)
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)
        b = self.bottleneck(self.pool2(x2), t_emb)
        u1 = self.up1(b)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.up_conv1(u1, t_emb)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.up_conv2(u2, t_emb)
        return self.final(u2)