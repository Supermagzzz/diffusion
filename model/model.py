import torch
from torch import nn
import math

M_REAL = 20
M = 6 + 6 * M_REAL
N = 5
IMG_N = 50
HIDDEN = 128
M_DIV = 1
BLOCKS = 1000
# [-0.5, 0.5] -> int

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
    def __init__(self, noise_level, device):
        super().__init__()
        self.device = device
        self.w_x = torch.rand((BLOCKS, HIDDEN), requires_grad=True).to(device)
        self.w_coords = torch.rand((HIDDEN * 6, HIDDEN), requires_grad=True).to(device)
        self.transformer = nn.Transformer(d_model=HIDDEN)
        self.make_coord_embed = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN * 6),
            nn.ReLU(),
            nn.Linear(HIDDEN * 6, HIDDEN * 6),
            nn.ReLU(),
            nn.Linear(HIDDEN * 6, HIDDEN * 6),
        )
        self.make_noise_result = nn.Sequential(
            nn.Linear(BLOCKS, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

        # self.make_coords = nn.Sequential(
        #     nn.Linear(HIDDEN, 2 * BLOCKS),
        #     nn.Softmax(dim=1),
        #     nn.Linear(2 * BLOCKS, 1),
        # )

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
        batch_size = svg.shape[0]
        svg = svg.reshape(batch_size, N * M)
        svg = torch.clamp((svg + 1) / 2 * BLOCKS, 0, BLOCKS - 1)
        coords = torch.zeros((batch_size, N * M // 6, 6, BLOCKS)).to(self.device)
        for b in range(batch_size):
            for i in range(N * M):
                coords[b][i // 6][i % 6][svg[b][i].long()] = 1
        coords = torch.matmul(coords, self.w_x)
        coords = coords.reshape(batch_size, N * M // 6, HIDDEN * 6)
        embeds = torch.matmul(coords, self.w_coords)
        noise_embeds = self.transformer(embeds, embeds)
        coord_embed = self.make_coord_embed(noise_embeds)
        coord_embed = coord_embed.reshape(batch_size, N * M, HIDDEN)
        bin_probs = torch.softmax(torch.matmul(coord_embed, self.w_x.permute(1, 0)), dim=-1)
        noise_result = self.make_noise_result(bin_probs)
        return noise_result.reshape(batch_size, N, M)

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
