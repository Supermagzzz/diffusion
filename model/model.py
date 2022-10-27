import torch
from torch import nn
import torch.nn.functional as F
import math

# M_REAL = 20
# M = 6 + 6 * M_REAL
# N = 5
# IMG_N = 50
# HIDDEN = 256
# M_DIV = 1
# BLOCKS = 2000000


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
    def __init__(self, common):
        super().__init__()
        self.range = 4
        self.common = common
        self.device = common.device
        self.make_time_embed = SinusoidalPositionEmbeddings(common.HIDDEN)
        self.w_x = nn.Parameter(torch.empty(common.BLOCKS, common.HIDDEN), requires_grad=True).to('cpu')
        self.w_coords = nn.Parameter(torch.normal(0, 1, (common.HIDDEN * 6, common.HIDDEN)), requires_grad=True)
        self.transformer = nn.Transformer(d_model=common.HIDDEN, dtype=torch.float, batch_first=True, num_encoder_layers=12, num_decoder_layers=12)
        self.make_coord_embed = nn.ModuleList([nn.Linear(common.HIDDEN, common.HIDDEN * 6)] + sum([[
            nn.Linear(common.HIDDEN * 6, common.HIDDEN * 6),
            nn.ReLU()
        ] for i in range(10)], []))
        self.make_noise_result = nn.ModuleList(sum([[
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.ReLU()
        ] for i in range(10)], []) + [nn.Linear(common.HIDDEN, 1)])

    def forward(self, svg, timestamp):
        batch_size = svg.shape[0]
        svg = svg.reshape(batch_size, self.common.N * self.common.M_REAL // 6, 6, 1)
        # coords = self.make_w_x(svg)
        svg = torch.clamp((svg + self.range) / (2 * self.range) * self.common.BLOCKS, 0, self.common.BLOCKS - 1).long()
        coords = F.embedding(svg, self.w_x).to(self.device)
        coords = coords.reshape(batch_size, self.common.N * self.common.M_REAL // 6, self.common.HIDDEN * 6)
        # embeds = self.make_w_coord(coords)
        embeds = torch.matmul(coords, self.w_coords)
        time_embed = self.make_time_embed(timestamp)
        time_embed = time_embed.reshape(batch_size, 1, self.common.HIDDEN)
        noise_embeds = self.transformer(torch.cat([time_embed, embeds], dim=1), embeds)
        # coord_embed = self.make_coord_embed(noise_embeds)
        coord_embed = noise_embeds
        for layer in self.make_coord_embed:
            coord_embed = layer(coord_embed)
        coord_embed = coord_embed.reshape(batch_size, self.common.N * self.common.M_REAL, self.common.HIDDEN)
        # noise_result = self.make_noise_result(coord_embed)
        noise_result = coord_embed
        for layer in self.make_noise_result:
            noise_result = layer(noise_result)
        return noise_result.reshape(batch_size, self.common.N, self.common.M_REAL)
