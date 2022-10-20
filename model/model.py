import torch
from torch import nn
import math

M_REAL = 20
M = 2 + 6 * M_REAL
N = 5
IMG_N = 50
HIDDEN = 128
M_DIV = 1


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


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
    def __init__(self):
        super().__init__()
        self.prepare = nn.Sequential(
            nn.Linear(2, HIDDEN),
            nn.ReLU()
        )
        self.transformer = nn.Transformer(d_model=HIDDEN)
        self.result = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 2)
        )
        # self.image_channels = 4 * N * ((M - 2) // 6) // M_DIV
        # self.input_sz = 64
        # self.hidden = 32
        #
        # image_channels = self.image_channels
        # input_sz = self.input_sz
        # down_channels = (64, 128, 256, 512, 1024)
        # up_channels = (1024, 512, 256, 128, 64)
        # out_dim = 1
        # time_emb_dim = 32
        #
        # self.time_mlp = nn.Sequential(
        #     SinusoidalPositionEmbeddings(time_emb_dim),
        #     nn.Linear(time_emb_dim, time_emb_dim),
        #     nn.ReLU()
        # )
        # self.prepareSvg = nn.Sequential(
        #     nn.Linear(N * M, input_sz * input_sz),
        #     nn.ReLU()
        # )
        #
        # self.conv0 = nn.Conv2d(image_channels + 1, down_channels[0], 3, padding=1)
        # self.downs = nn.ModuleList(
        #     [Block(down_channels[i], down_channels[i + 1], time_emb_dim) for i in range(len(down_channels) - 1)]
        # )
        # self.ups = nn.ModuleList(
        #     [Block(up_channels[i], up_channels[i + 1], time_emb_dim, True) for i in range(len(up_channels) - 1)]
        # )
        # self.output = nn.Sequential(
        #     nn.Conv2d(up_channels[-1], image_channels, out_dim),
        #     nn.ReLU()
        # )
        #
        # self.svgLinear = nn.Sequential(
        #     nn.Linear(N * M, 4 * N * M),
        #     nn.ReLU()
        # )
        #
        # self.pngBigger = nn.Sequential(
        #     nn.Linear(M_REAL // M_DIV, M)
        # )
        # self.pngLinear = nn.Sequential(
        #     nn.Linear(input_sz * input_sz, self.hidden),
        #     nn.ReLU()
        # )
        #
        # self.linearWithSvg = nn.Sequential(
        #     nn.Linear(4 * N * M * (self.hidden + 1), 4 * N * M),
        #     nn.ReLU(),
        #     nn.Linear(4 * N * M, N * M),
        #     nn.ReLU(),
        #     nn.Linear(N * M, N * M)
        # )

    def forward(self, svg, timestep):
        svg = svg.reshape(-1, N * M // 2, 2)
        svg = self.prepare(svg)
        svg = self.transformer(svg, svg)
        svg = self.result(svg)
        svg = svg.reshape(-1, N, M)
        return svg
        # t = self.time_mlp(timestep)
        # prep_svg = self.prepareSvg(svg.reshape(-1, N * M))
        # prep_svg = prep_svg.reshape(-1, self.input_sz, self.input_sz, 1)
        # x = torch.cat((x, prep_svg), dim=-1)
        # x = x.permute(0, 3, 1, 2)
        # x = self.conv0(x)
        # residual_inputs = []
        # for down in self.downs:
        #     x = down(x, t)
        #     residual_inputs.append(x)
        # for up in self.ups:
        #     residual_x = residual_inputs.pop()
        #     x = torch.cat((x, residual_x), dim=1)
        #     x = up(x, t)
        # x = self.output(x)
        # x = x.reshape(-1, x.shape[1], x.shape[2] * x.shape[3])
        #
        # svg = self.svgLinear(svg.reshape(-1, N * M)).reshape(-1, 4 * N * M, 1)
        #
        # png = x.reshape(-1, 4 * N, M_REAL // M_DIV, self.input_sz * self.input_sz)
        # png = png.permute(0, 1, 3, 2)
        # png = self.pngBigger(png)
        # png = png.permute(0, 1, 3, 2)
        # png = self.pngLinear(png)
        # png = png.reshape(-1, 4 * N * M, self.hidden)
        # all = torch.cat([svg, png], dim=-1).reshape(-1, 4 * N * M * (self.hidden + 1))
        #
        # return self.linearWithSvg(all).reshape(-1, N, M)
