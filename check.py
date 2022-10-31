from deepsvg.svglib.svg import SVG

import torch
from torch import nn
import matplotlib.pyplot as plt
from dataset.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
from model.model import SimpleDenoiser
from common import Common
import imageio.v2 as imageio

torch.set_default_dtype(torch.float32)
common = Common(check=True)

model = SimpleDenoiser(common)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('model_weights'))
model.to(common.device)
print(common.device)

def print_example(data):
    plt.figure(figsize=(15, 3))
    for i in range(len(data)):
        path = 'tmp/check_' + str(i) + '.png'
        common.make_svg(data[i]).save_png(path)
        im = imageio.imread(path)
        plt.subplot(1, len(data), i + 1)
        plt.imshow(im)
    plt.savefig('trash/check')

img = torch.randn((1, common.N, common.M_REAL), device=common.device)
step = common.T // 10
data = []
for i in range(common.T - 1, -1, -1):
    for j in range(3):
        t = torch.full((1,), i, device=common.device, dtype=torch.long)
        img = common.sample_timestep(img, t, model(img, t))
        if i % step == 0 and j == 0:
            data.append(img[0])
print_example(data)
torch.save(model.state_dict(), 'model_weights')
print('saved')
exit(0)

for batch in common.dataloader:
    batch = torch.cat([batch[:, :, :2], batch[:, :, :2], batch], dim=-1)
    batch = batch.to(common.device)
    t = torch.zeros(batch.shape[0]).long().to(common.device)
    noised, noise = common.forward_diffusion_sample(batch, t)
    pred_noise = model(noised, t)

    loss = common.calc_loss(noise, pred_noise)
    baseline = common.calc_loss(noise, -batch * common.know_level)
    print(loss.item(), baseline.item(), (loss / baseline).item())
    for x in range(common.apply_batch_sz):
        common.make_svg(common.sample_timestep(noised[x], torch.Tensor([t[x]]).long().to(common.device), pred_noise[x])).save_svg('trash/' + 'test' + str(x) + '.svg')
        common.make_svg(batch[x]).save_svg('trash/' + 'real' + str(x) + '.svg')
        common.make_svg(noised[x]).save_svg('trash/' + 'inp' + str(x) + '.svg')
    break

