from deepsvg.svglib.svg import SVG

import torch
from torch import nn

from dataset.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
from model.model import SimpleDenoiser
from common import Common

torch.set_default_dtype(torch.float32)
common = Common(check=True)

model = SimpleDenoiser(common)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('model_weights'))
model.to(common.device)
print(common.device)

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
        common.make_svg(common.sample_timestep(noised, t, pred_noise)).save_svg('trash/' + 'test' + str(x) + '.svg')
        common.make_svg(batch[x]).save_svg('trash/' + 'real' + str(x) + '.svg')
        common.make_svg(noised[x]).save_svg('trash/' + 'inp' + str(x) + '.svg')

