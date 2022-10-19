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

        self.dense = nn.Linear(
            embed_in,
            embed_out
        )
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.dropout(self.dense(x)))


class SimpleDenoiser(nn.Module):

    def __init__(self):
        super().__init__()
        self.config = [256, 256, 512, 512, 1024, 2048]
        self.down = []
        self.down.append(Block(N * M, self.config[0]))
        for i in range(1, len(self.config)):
            self.down.append(Block(self.config[i - 1], self.config[i]))
        self.up = [None for i in self.down]
        for i in range(len(self.config) - 1, 0, -1):
            self.up[i] = Block(self.config[i], self.config[i - 1])
        self.linear = nn.Linear(self.config[0], N * M)

    def forward(self, image, timestep):
        image = image.view(-1, N * M)
        results = [None for i in self.config]
        results[0] = self.down[0](image)
        for i in range(1, len(self.config)):
            results[i] = self.down[i](results[i - 1])
        output = results[-1]
        for i in range(len(self.config) - 1, 0, -1):
            output = self.up[i](output + results[i])
        output = self.linear(output)
        return output.reshape(-1, N, M)


