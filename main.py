import torch
from torch import nn
from model.model import SimpleDenoiser
from common import Common
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


T = 300
betas = linear_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


common = Common()

model = SimpleDenoiser(common.device)
model = nn.DataParallel(model)
model.to(common.device)
print(common.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

@torch.no_grad()
def sample_timestep(x, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def print_example(data, index):
    plt.figure(figsize=(15, 3))
    for i in range(len(data)):
        path = 'tmp/' + str(i) + '.png'
        common.make_svg(data[i]).save_png(path)
        im = imageio.imread(path)
        plt.subplot(1, len(data), i + 1)
        plt.imshow(im)
    plt.savefig('trash/plt' + str(index))


for epoch in range(100000):
    for step, batch in enumerate(common.dataloader):
        real = common.make_sample(batch)
        optimizer.zero_grad()

        t = torch.randint(0, T, (batch.shape[0],), device=common.device).long()
        noised, noise = forward_diffusion_sample(real, t, common.device)

        pred_noise = model(noised, t)

        loss = common.calc_loss(noise, pred_noise)
        baseline = common.calc_loss(noise, -real * common.know_level)

        loss.backward()
        optimizer.step()
        print(epoch, loss.item(), baseline.item(), (loss / baseline).item())

    if epoch % 100 == 0:
        img = torch.randn((1, common.N, common.M_REAL), device=common.device)
        step = T // 10
        data = []
        for i in range(T - 1, -1, -1):
            t = torch.full((1,), i, device=common.device, dtype=torch.long)
            img = sample_timestep(img, t)
            if i % step == 0:
                data.append(img[0])
        print_example(data, epoch // 100)
        torch.save(model.state_dict(), 'model_weights')
        print('saved')

