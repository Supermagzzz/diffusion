import torch
from torch import nn
import torch.nn.functional as F
import math

import common


class SimpleDenoiser(nn.Module):
    def __init__(self, common):
        super().__init__()
        self.range = 4
        self.common = common
        self.device = common.device
        self.time_embed_table = nn.Parameter(torch.normal(common.T, common.HIDDEN), requires_grad=True)
        self.w_x = nn.Parameter(torch.normal(common.BLOCKS, common.HIDDEN), requires_grad=True)
        self.w_coords = nn.Linear(common.HIDDEN * 6, common.HIDDEN)
        self.unite_with_time = nn.Sequential(
            nn.Linear(common.HIDDEN * 2, common.HIDDEN),
            nn.ReLU(),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.ReLU(),
            nn.Linear(common.HIDDEN, common.HIDDEN),
        )
        self.transformer = nn.Transformer(d_model=common.HIDDEN, dtype=torch.float, batch_first=True, num_encoder_layers=12, num_decoder_layers=12)
        self.make_coord_embed = nn.ModuleList([nn.Linear(common.HIDDEN, common.HIDDEN * 6)] + sum([[
            nn.Linear(common.HIDDEN * 6, common.HIDDEN * 6),
            nn.ReLU()
        ] for i in range(4)], []) + [nn.Linear(common.HIDDEN * 6, common.HIDDEN * 6)])
        self.make_noise_result = nn.ModuleList(sum([[
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.ReLU()
        ] for i in range(4)], []) + [nn.Linear(common.HIDDEN, 1)])

    def forward(self, svg, timestamp):
        batch_size = svg.shape[0]
        svg = svg.reshape(batch_size, self.common.N * self.common.M_REAL // 6, 6, 1)
        # coords = self.make_w_x(svg)
        svg = torch.clamp((svg + self.range) / (2 * self.range) * self.common.BLOCKS, 0, self.common.BLOCKS - 1).long()
        coords = F.embedding(svg, self.w_x).to(self.device)
        coords = coords.reshape(batch_size, self.common.N * self.common.M_REAL // 6, self.common.HIDDEN * 6)
        # embeds = self.make_w_coord(coords)
        embeds = self.w_coords(coords)
        time_embed = F.embedding(timestamp, self.time_embed_table)
        time_embed = time_embed.reshape(batch_size, 1, self.common.HIDDEN)
        time_embed = torch.cat([time_embed for x in range(embeds.shape[1])], dim=1)
        embeds = self.unite_with_time(torch.cat([embeds, time_embed], dim=-1))
        noise_embeds = self.transformer(embeds, embeds)
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
