from deepsvg.svglib.svg import SVG

import torch

from dataset.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
from model.model import SimpleDenoiser
from common import Common

torch.set_default_dtype(torch.float32)
common = Common(check=True)

model = SimpleDenoiser(common.device)
model.load_state_dict(torch.load('model_weights'))
model.to(common.device)
print(common.device)

batch = next(common.dataloader)
batch = torch.cat([batch[:, :, :2], batch[:, :, :2], batch], dim=-1)
batch = batch.to(common.device)
noise = common.add_noise(batch, common.noise_level).to(common.device)
new_img = batch + noise
pred_noise = model(new_img, torch.Tensor(1).to(common.device))

loss = common.calc_loss(noise, pred_noise)
baseline = common.calc_loss(noise, -batch * common.know_level)
print(loss.item(), baseline.item(), (loss / baseline).item())
for x in range(common.apply_batch_sz):
    common.make_svg(new_img[x] - pred_noise[x]).save_svg('trash/' + 'test' + str(x) + '.svg')
    common.make_svg(batch[x]).save_svg('trash/' + 'real' + str(x) + '.svg')
    common.make_svg(new_img[x]).save_svg('trash/' + 'inp' + str(x) + '.svg')

