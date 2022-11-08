import torch
from torch import nn
import torch.nn.functional as F
import math

import common


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
        self.time_embed_table = nn.Parameter(torch.empty((common.T, common.HIDDEN)), requires_grad=True)
        self.pos_embed_table = nn.Sequential(
            SinusoidalPositionEmbeddings(common.HIDDEN),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.ReLU()
        )
        self.w_x = nn.Parameter(torch.normal(0, 1, (common.BLOCKS, common.HIDDEN)), requires_grad=True)
        self.unite_with_real_svg = nn.Sequential(
            nn.Linear(common.HIDDEN + 1, common.HIDDEN),
            nn.Tanh(),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.ReLU(),
            nn.Linear(common.HIDDEN, common.HIDDEN)
        )
        self.w_coords = nn.Linear(common.HIDDEN * 6, common.HIDDEN)
        self.unite_with_embeds = nn.Sequential(
            nn.Linear(common.HIDDEN * 3, common.HIDDEN),
            nn.Tanh(),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.ReLU(),
            nn.Linear(common.HIDDEN, common.HIDDEN),
        )
        self.transformer = nn.Transformer(d_model=common.HIDDEN, dtype=torch.float, batch_first=True, num_encoder_layers=12, num_decoder_layers=12)
        self.make_coord_embed = nn.ModuleList([nn.Linear(common.HIDDEN * 2, common.HIDDEN * 6)] + sum([[
            nn.Linear(common.HIDDEN * 6, common.HIDDEN * 6),
            nn.Tanh()
        ] for i in range(3)], []) + [nn.Linear(common.HIDDEN * 6, common.HIDDEN * 6)])
        self.make_noise_result = nn.ModuleList(sum([[
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.ReLU()
        ] for i in range(3)], []) + [nn.Linear(common.HIDDEN, 1)])

    def forward(self, svg, timestamp):
        batch_size = svg.shape[0]
        svg = svg.reshape(batch_size, self.common.N * self.common.M_REAL // 6, 6, 1)
        # coords = self.make_w_x(svg)
        svg_long = torch.clamp((svg + self.range) / (2 * self.range) * self.common.BLOCKS, 0, self.common.BLOCKS - 1).long()
        coords = F.embedding(svg_long, self.w_x).to(self.device)
        coords = self.unite_with_real_svg(torch.cat([coords, svg], dim=-1))
        coords = coords.reshape(batch_size, self.common.N * self.common.M_REAL // 6, self.common.HIDDEN * 6)
        # embeds = self.make_w_coord(coords)
        embeds = self.w_coords(coords)

        time_embed = F.embedding(timestamp, self.time_embed_table)
        time_embed = time_embed.reshape(batch_size, 1, self.common.HIDDEN)
        time_embed = torch.cat([time_embed for x in range(embeds.shape[1])], dim=1)

        pos_embed = torch.Tensor([i for i in range(embeds.shape[1])]).long().to(self.device)
        pos_embed = self.pos_embed_table(pos_embed)
        pos_embed = torch.stack([pos_embed for i in range(batch_size)])

        embeds = self.unite_with_embeds(torch.cat([embeds, time_embed, pos_embed], dim=-1))
        noise_embeds = self.transformer(embeds, embeds)
        # coord_embed = self.make_coord_embed(noise_embeds)
        coord_embed = torch.cat([noise_embeds, embeds], dim=-1)
        for layer in self.make_coord_embed:
            coord_embed = layer(coord_embed)
        coord_embed = coord_embed.reshape(batch_size, self.common.N * self.common.M_REAL, self.common.HIDDEN)
        # noise_result = self.make_noise_result(coord_embed)
        noise_result = coord_embed
        for layer in self.make_noise_result:
            noise_result = layer(noise_result)
        return noise_result.reshape(batch_size, self.common.N, self.common.M_REAL)
