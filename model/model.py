import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Config
import math


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
        self.add_time_embed_table = nn.Embedding(common.T, common.HIDDEN)
        self.get_time_embed_table_normal = nn.Embedding(common.T, common.HIDDEN)
        self.get_time_embed_table_sinus = SinusoidalPositionEmbeddings(common.HIDDEN)
        self.get_time_embed_table = nn.Sequential(
            nn.Linear(common.HIDDEN * 3, common.HIDDEN),
            nn.ReLU(),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.ReLU(),
            nn.Linear(common.HIDDEN, common.HIDDEN)
        )
        self.pos_embed_table = nn.Sequential(
            SinusoidalPositionEmbeddings(common.HIDDEN),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.ReLU()
        )
        self.w_x = nn.Embedding(common.BLOCKS, common.HIDDEN).to('cpu')
        self.unite_with_real_svg = nn.Sequential(
            nn.Linear(common.HIDDEN + 2, common.HIDDEN),
            nn.ReLU(),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.ReLU(),
            nn.Linear(common.HIDDEN, common.HIDDEN)
        )
        self.w_coords = nn.Sequential(
            nn.Linear(common.HIDDEN * 6, common.HIDDEN),
            nn.ReLU()
        )
        self.unite_with_embeds = nn.Sequential(
            nn.Linear(common.HIDDEN * 4, common.HIDDEN),
            nn.ReLU(),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.ReLU(),
            nn.Linear(common.HIDDEN, common.HIDDEN),
        )
        self.transformer = nn.Transformer(d_model=common.HIDDEN, dtype=torch.float, batch_first=True)
        # self.transformer = T5ForConditionalGeneration(T5Config(common.BLOCKS))
        # self.make_probs = nn.Sequential(
        #     nn.Softmax(),
        #     nn.Linear(common.BLOCKS, common.HIDDEN),
        #     nn.ReLU()
        # )
        self.make_coord_embed = nn.ModuleList([nn.Linear(common.HIDDEN * 2, common.HIDDEN * 6), nn.Tanh()] + sum([[
            nn.Linear(common.HIDDEN * 6, common.HIDDEN * 6),
            nn.ReLU()
        ] for i in range(2)], []) + [nn.Linear(common.HIDDEN * 6, common.HIDDEN * 6), nn.ReLU()])
        self.make_noise_result = nn.ModuleList([nn.Linear(common.HIDDEN * 3, common.HIDDEN), nn.Tanh()] + sum([[
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.ReLU()
        ] for i in range(2)], []) + [nn.Linear(common.HIDDEN, 1)])

    def make_seq(self, data, embeds):
        data = data.reshape(embeds.shape[0], 1, self.common.HIDDEN)
        return torch.cat([data for x in range(embeds.shape[1])], dim=1)

    def forward(self, svg, timestamp):
        batch_size = svg.shape[0]
        svg = svg.reshape(batch_size, self.common.N * self.common.M_REAL // 6, 6)
        svg_long = torch.clamp((svg + self.range) / (2 * self.range) * self.common.BLOCKS, 0, self.common.BLOCKS - 1).long()
        svg_rem = torch.fmod((svg + self.range) / (2 * self.range) * self.common.BLOCKS, 1)
        coords = self.w_x(svg_long).to(self.device)
        svg = svg.reshape(batch_size, self.common.N * self.common.M_REAL // 6, 6, 1)
        svg_rem = svg_rem.reshape(batch_size, self.common.N * self.common.M_REAL // 6, 6, 1)
        coords = self.unite_with_real_svg(torch.cat([coords, svg, svg_rem], dim=-1))
        coords = coords.reshape(batch_size, self.common.N * self.common.M_REAL // 6, self.common.HIDDEN * 6)
        embeds = self.w_coords(coords)

        time_embed = self.make_seq(self.add_time_embed_table(timestamp), embeds)

        pos_embed = torch.Tensor([i for i in range(embeds.shape[1])]).long().to(self.device)
        pos_embed = self.pos_embed_table(pos_embed)
        pos_embed = torch.stack([pos_embed for i in range(batch_size)])

        embeds = self.unite_with_embeds(torch.cat([embeds, time_embed, pos_embed], dim=-1))

        out_embeds = self.get_time_embed_table(torch.cat([
            self.make_seq(self.get_time_embed_table_normal(timestamp), embeds),
            self.make_seq(self.get_time_embed_table_sinus(timestamp), embeds),
            embeds,
            pos_embed
        ], dim=-1))

        svg = svg.reshape(batch_size, -1)
        svg_long = torch.clamp((svg + self.range) / (2 * self.range) * self.common.BLOCKS, 0, self.common.BLOCKS - 1).long()
        noise_embeds = self.transformer(embeds, out_embeds)
        coord_embed = torch.cat([noise_embeds, noise_embeds], dim=-1)
        for layer in self.make_coord_embed:
            coord_embed = layer(coord_embed)
        coord_embed = coord_embed.reshape(batch_size, self.common.N * self.common.M_REAL, self.common.HIDDEN)

        # prob_embeds = self.make_probs(torch.matmul(self.w_x, coord_embed.permute(0, 2, 1)).permute(0, 2, 1))
        # coord_prob_embeds = coord_embed * prob_embeds
        # noise_result = torch.cat([coord_embed, prob_embeds, coord_prob_embeds], dim=-1)
        noise_result = torch.cat([coord_embed, coord_embed, coord_embed], dim=-1)
        for layer in self.make_noise_result:
            noise_result = layer(noise_result)
        return noise_result.reshape(batch_size, self.common.N, self.common.M_REAL)
