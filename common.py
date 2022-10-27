import torch

from dataset.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from deepsvg.svglib.svg import SVG


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class Common:
    def __init__(self, check=False):
        self.N = 5
        self.M = 8
        self.HIDDEN = 256
        self.BLOCKS = 2000000
        self.M_REAL = self.M * 6 + 6
        self.noise_level = 0.01
        self.know_level = 0.01
        self.batch_sz = 512
        self.cpu_batch_sz = 1
        self.apply_batch_sz = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = CustomImageDataset('data/tensors')

        if check:
            self.real_batch_sz = self.apply_batch_sz
        else:
            self.real_batch_sz = self.cpu_batch_sz if self.device == "cpu" else self.batch_sz

        self.dataloader = DataLoader(self.dataset, self.real_batch_sz, shuffle=False, drop_last=True)

        self.T = 300
        self.betas = linear_beta_schedule(timesteps=self.T)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def calc_loss(self, a, b):
        # return F.l1_loss(a, b)
        return (a - b).pow(2).sum()

    def make_sample(self, batch):
        batch = torch.cat([batch[:, :, :2], batch[:, :, :2], batch], dim=-1)
        return batch.to(self.device)

    def add_noise(self, tensor, mult):
        return (torch.normal(0, mult, size=tensor.shape).to(self.device) - tensor * self.know_level).to(self.device)

    def forward_diffusion_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) \
            + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)

    @torch.no_grad()
    def sample_timestep(self, x, t, predict):
        betas_t = get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * predict / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def make_svg(self, tensor):
        tensor += 0.7
        tensor *= 17
        data = []
        for row in tensor:
            command = [-1] * 14
            command[0] = 0
            command[-2] = row[0]
            command[-1] = row[1]
            lastX, lastY = row[0], row[1]
            data.append(command)
            for i in range(6, row.shape[0], 6):
                command = [-1] * 14
                command[0] = 2
                command[-8] = lastX
                command[-7] = lastY
                command[-6] = row[i]
                command[-5] = row[i + 1]
                command[-4] = row[i + 2]
                command[-3] = row[i + 3]
                command[-2] = row[i + 4]
                command[-1] = row[i + 5]
                lastX, lastY = command[-2], command[-1]
                data.append(command)
        return SVG.from_tensor(torch.Tensor(data))

