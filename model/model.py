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


class SimpleDenoiser(nn.Module):

    def __init__(self):
        super().__init__()

        self.simple = nn.Sequential(
            nn.Linear(N * M, N * M),
            nn.ReLU(),
            nn.Linear(N * M, N * M),
            nn.ReLU(),
            nn.Linear(N * M, N * M),
        )

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
            nn.Linear(N * HIDDEN, N * M)
        )

    def forward(self, image, timestep):
        image = image.view(-1, N * M)
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

