import math
import sys

from deepsvg.svglib.svg import SVG

import pydiffvg
import torch
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.functional import l1_loss

from dataset.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
from model.model import SimpleDenoiser

torch.set_default_dtype(torch.float32)
noise_level = 0.03
know_level = 0.01
batch_sz = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CustomImageDataset('data/tensors')
dataloader = DataLoader(dataset, batch_size=batch_sz if device == "cpu" else batch_sz, shuffle=False, drop_last=True)


def add_noise(tensor, mult):
    return torch.normal(0, mult, size=tensor.shape).to(device) - tensor * know_level


def make_svg(tensor):
    tensor -= torch.min(tensor) - 0.5
    tensor *= 10
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


N = 5
M = 8

model = SimpleDenoiser(noise_level, device)
model.load_state_dict(torch.load('model_weights'))
model.to(device)
print(device)

for step, batch in enumerate(dataloader):
    batch = torch.cat([batch[:, :, :2], batch[:, :, :2], batch], dim=-1)
    batch = batch.to(device)
    noise = add_noise(batch, noise_level).to(device)
    new_img = batch + noise
    pred_noise = model(noise, torch.Tensor(1).to(device))

    def gloss(a, b):
        return (a - b).pow(2).sum()

    loss = gloss(noise, pred_noise)
    baseline = gloss(noise, -batch * know_level)
    print(step, loss.item(), baseline.item(), (loss / baseline).item())
    for x in range(batch_sz):
        make_svg(new_img[x] - pred_noise[x]).save_svg('trash/' + 'test' + str(x) + '.svg')
        make_svg(batch[x]).save_svg('trash/' + 'real' + str(x) + '.svg')
        make_svg(new_img[x]).save_svg('trash/' + 'inp' + str(x) + '.svg')
    exit(0)

