import torch
from torch import nn
import math

M = 2 + 6 * 10
N = 5
HIDDEN = 128


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


class Block(nn.Module):
    def __init__(self, embed_in, embed_out):
        super().__init__()

        self.dense1 = nn.Linear(
            embed_in,
            embed_out // 2
        )
        self.dense2 = nn.Linear(
            embed_in,
            embed_out // 2
        )
        self.dense3 = nn.Linear(
            embed_in,
            embed_out // 2
        )
        self.relu1 = nn.Tanh()
        self.dense4 = nn.Linear(
            embed_out,
            embed_out
        )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x1 = self.dense1(x)
        x2 = self.dense2(x)
        x3 = self.dense3(x)
        dot = torch.mul(x2, x3)
        res = torch.cat([x1, dot], dim=-1)
        res = self.relu1(res)
        res = self.dense4(res)
        return self.relu2(res)


class SimpleDenoiser(nn.Module):

    def __init__(self):
        super().__init__()

        self.funny = nn.Sequential(
            Block(N * M, N * HIDDEN),
            Block(N * HIDDEN, N * HIDDEN),
            # Block(N * HIDDEN, N * HIDDEN),
            # Block(N * HIDDEN, N * HIDDEN),
            Block(N * HIDDEN, N * M)
        )

        self.final_linear = nn.Sequential(
            nn.Linear(2 * N * M, N * M)
        )
        return

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(N * HIDDEN),
            nn.Linear(N * HIDDEN, N * HIDDEN),
            nn.ReLU()
        )

        self.prepare_embed = nn.Sequential(
            nn.Linear(N * M, N * HIDDEN),
            nn.ReLU(),
            nn.Linear(N * HIDDEN, N * HIDDEN),
            nn.ReLU(),
        )

        self.unite_time = nn.Sequential(
            nn.Linear(2 * N * HIDDEN, 2 * N * HIDDEN),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.3)

        self.query = nn.Sequential(
            nn.Linear(2 * N * HIDDEN, N * HIDDEN),
            nn.ReLU()
        )

        self.key = nn.Sequential(
            nn.Linear(2 * N * HIDDEN, N * HIDDEN),
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(2 * N * HIDDEN, N * HIDDEN),
            nn.ReLU()
        )

        self.attn = nn.MultiheadAttention(HIDDEN, 1, batch_first=True)
        self.attnReLu = nn.ReLU()

        self.result = nn.Sequential(
            nn.Linear(N * HIDDEN, N * HIDDEN),
            nn.ReLU(),
            nn.Linear(N * HIDDEN, N * M),
            nn.Tanh(),
            nn.Linear(N * M, N * M),
        )

    def forward(self, image, timestep):
        image = image.view(-1, N * M)
        return self.final_linear(torch.cat([self.funny(image), image], dim=1)).reshape(-1, N, M)
        t = self.time_mlp(timestep)
        image = self.prepare_embed(image)
        batch_size = image.shape[0]
        t = t.repeat(batch_size, 1)
        united = self.unite_time(torch.cat([t, image], dim=1))
        united = self.dropout(united)
        query = self.query(united).view(-1, N, HIDDEN)
        key = self.key(united).view(-1, N, HIDDEN)
        value = self.value(united).view(-1, N, HIDDEN)
        attn_out, attn_weights = self.attn(query, key, value)
        attn_out = self.attnReLu(attn_out)
        attn_out = attn_out.reshape(-1, N * HIDDEN)
        res = self.result(attn_out)
        return res.reshape(-1, N, M)

