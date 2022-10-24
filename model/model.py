import torch
from torch import nn
import torch.nn.functional as F
import math

M_REAL = 8
M = 6 + 6 * M_REAL
N = 5
IMG_N = 50
HIDDEN = 128
M_DIV = 1
BLOCKS = 1000000


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(1000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleDenoiser(nn.Module):
    def __init__(self, noise_level, device):
        super().__init__()
        self.device = device
        self.w_x = nn.Parameter(torch.normal(0, 1, (BLOCKS * 6, HIDDEN)), requires_grad=True).to('cpu')
        self.w_coords = nn.Parameter(torch.normal(0, 1, (HIDDEN * 6, HIDDEN)), requires_grad=True)
        self.transformer = nn.Transformer(d_model=HIDDEN, dtype=torch.float)
        self.make_coord_embed = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN * 6).to(device),
            nn.ReLU().to(device),
            nn.Linear(HIDDEN * 6, HIDDEN * 6).to(device),
            nn.ReLU().to(device),
            nn.Linear(HIDDEN * 6, HIDDEN * 6).to(device),
        )
        self.make_noise_result = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN).to(device),
            nn.ReLU().to(device),
            nn.Linear(HIDDEN, HIDDEN).to(device),
            nn.Tanh().to(device),
            nn.Linear(HIDDEN, 1).to(device)
        )

    def forward(self, svg, timestep):
        batch_size = svg.shape[0]
        svg = svg.reshape(batch_size, N * M // 6, 6)
        svg = torch.clamp((svg + 1) / 2 * BLOCKS, 0, BLOCKS - 1).long()
        coords = F.embedding(svg, self.w_x).to(self.device)
        coords = coords.reshape(batch_size, N * M // 6, HIDDEN * 6)
        embeds = torch.matmul(coords, self.w_coords)
        noise_embeds = self.transformer(embeds, embeds)
        coord_embed = self.make_coord_embed(noise_embeds)
        coord_embed = coord_embed.reshape(batch_size, N * M, HIDDEN)
        # bin_probs = torch.softmax(torch.matmul(coord_embed.to('cpu'), self.w_x.permute(1, 0)), dim=-1).to(self.device)
        noise_result = self.make_noise_result(coord_embed)
        return noise_result.reshape(batch_size, N, M)
