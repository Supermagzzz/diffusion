import torch
from torch import nn
import torch.nn.functional as F
import math

M_REAL = 20
M = 6 + 6 * M_REAL
N = 5
IMG_N = 50
HIDDEN = 256
M_DIV = 1
BLOCKS = 2000000


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
    def __init__(self, device):
        super().__init__()
        self.range = 4
        self.device = device
        self.make_time_embed = SinusoidalPositionEmbeddings(HIDDEN)
        self.make_w_x = self.make_w_x = nn.Sequential(
            nn.Linear(1, BLOCKS),
            nn.ReLU(),
            nn.Linear(BLOCKS, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN)
        )
        self.make_w_coord = nn.Linear(
            HIDDEN * 6, HIDDEN
        )
        # self.w_x = nn.Parameter(torch.empty(BLOCKS, HIDDEN), requires_grad=True).to('cpu')
        # self.w_coords = nn.Parameter(torch.normal(0, 1, (HIDDEN * 6, HIDDEN)), requires_grad=True)
        self.transformer = nn.Transformer(d_model=HIDDEN, dtype=torch.float, batch_first=True)
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

    def forward(self, svg, timestamp):
        batch_size = svg.shape[0]
        svg = svg.reshape(batch_size, N * M // 6, 6)
        coords = self.make_w_x(svg)
        # svg = torch.clamp((svg + self.range) / (2 * self.range) * BLOCKS, 0, BLOCKS - 1).long()
        # coords = F.embedding(svg, self.w_x).to(self.device)
        coords = coords.reshape(batch_size, N * M // 6, HIDDEN * 6)
        embeds = self.make_w_coord(coords)
        # embeds = torch.matmul(coords, self.w_coords)
        time_embed = self.make_time_embed(timestamp)
        time_embed = time_embed.reshape(batch_size, 1, HIDDEN)
        noise_embeds = self.transformer(torch.cat([time_embed, embeds], dim=1), embeds)
        coord_embed = self.make_coord_embed(noise_embeds)
        coord_embed = coord_embed.reshape(batch_size, N * M, HIDDEN)
        noise_result = self.make_noise_result(coord_embed)
        return noise_result.reshape(batch_size, N, M)
