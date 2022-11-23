import torch
from torch import nn
from transformers import BertConfig, BertModel
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


class SimpleDiscriminator(nn.Module):
    def __init__(self, common):
        super().__init__()

        self.range = 4
        self.common = common
        self.device = common.device

        self.pos_embed_learned_table = nn.Embedding(self.common.N * self.common.M_REAL // 6, common.HIDDEN)
        self.pos_embed_table = nn.Sequential(
            SinusoidalPositionEmbeddings(common.HIDDEN),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.LeakyReLU(0.1)
        )
        self.unite_pos = nn.Sequential(
            nn.Linear(common.HIDDEN * 2, common.HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(common.HIDDEN, common.HIDDEN),
        )

        self.w_x = nn.Embedding(common.BLOCKS, common.HIDDEN)

        self.w_coords = nn.Sequential(
            nn.Linear(common.HIDDEN * 6, common.HIDDEN),
            nn.LeakyReLU(0.1)
        )
        self.make_in_embed = nn.Sequential(
            nn.Linear(common.HIDDEN * 2, common.HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(common.HIDDEN, common.HIDDEN),
        )

        self.make_out_embed = nn.Sequential(
            nn.Linear(common.HIDDEN * 2, common.HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(common.HIDDEN, common.HIDDEN)
        )

        self.cls_token = nn.Embedding(1, common.HIDDEN)
        self.encoder = BertModel(BertConfig(common.BLOCKS, common.HIDDEN, num_attention_heads=8, num_hidden_layers=12,
                                            max_position_embeddings=self.common.N * self.common.M_REAL // 6 + 1))

        self.transform_embed = nn.Sequential(
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(common.HIDDEN, 1),
        )

    def make_seq(self, data, embeds):
        data = data.reshape(embeds.shape[0], 1, self.common.HIDDEN)
        return torch.cat([data for x in range(embeds.shape[1])], dim=1)

    def make_batch(self, data, batch_size):
        return torch.stack([data for x in range(batch_size)])

    def forward(self, svg):
        batch_size = svg.shape[0]

        svg = svg.reshape(batch_size, self.common.N * self.common.M_REAL)
        svg_long = torch.clamp((svg + self.range) / (2 * self.range) * self.common.BLOCKS, 0,
                               self.common.BLOCKS - 1).long()
        encoded_coords = self.w_x(svg_long)

        coords = encoded_coords.reshape(batch_size, self.common.N * self.common.M_REAL // 6, self.common.HIDDEN * 6)

        embeds = self.w_coords(coords)

        pos_embed = torch.Tensor([i for i in range(embeds.shape[1])]).long().to(self.device)
        pos_embed = self.make_batch(self.unite_pos(
            torch.cat([self.pos_embed_table(pos_embed), self.pos_embed_learned_table(pos_embed)], dim=-1)), batch_size)

        embeds = self.make_in_embed(torch.cat([
            embeds,
            pos_embed
        ], dim=-1))

        cls_token = self.make_batch(self.cls_token(torch.LongTensor([0]).to(self.device)), batch_size)
        latent_embed = self.encoder(inputs_embeds=torch.cat([cls_token, embeds], dim=1)).last_hidden_state[:, 0, :]

        return self.transform_embed(latent_embed).reshape(batch_size)
