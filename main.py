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

torch.set_default_dtype(torch.float)
noise_level = 0.03
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CustomImageDataset('data/tensors')
dataloader = DataLoader(dataset, batch_size=1 if device == "cpu" else 128, shuffle=False, drop_last=True)


def add_noise(tensor, mult):
    return torch.normal(0, mult, size=tensor.shape).to(device) - tensor * 0.01

N = 5
M = 20

def make_image(inp):
    # return el
    canvas_width, canvas_height = 64, 64
    el = (inp + 0.5)
    el *= int(0.8 * canvas_width)
    el += int(0.1 * canvas_width)

    all_ims = []
    pathes = []
    groups = []
    for j, row in enumerate(el):
        for i in range(6, row.shape[0], 6):
            num_control_points = []
            points = []
            num_control_points.append(2)
            points.append(i - 2)
            points.append(i - 1)
            points.append(i + 2)
            points.append(i + 3)
            points.append(i)
            points.append(i + 1)
            points.append(i + 4)
            points.append(i + 5)
            points = row[points].reshape(-1, 2)
        pathes.append(pydiffvg.Path(torch.Tensor(num_control_points).to(device), points, False))
        groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(groups)]).to(device), fill_color=None,
                                          stroke_color=torch.Tensor([0, 0, 0, 1]).to(device)))
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, pathes, groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width,  # width
                 canvas_height,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 1,  # seed
                 None,
                 *scene_args)
    return img


model = SimpleDenoiser(noise_level, device)
print(device)
model.to(device)
optimizer = Adam(model.parameters(), lr=0.0001)
new_img, noise = None, None
for epoch in range(100000):
    for step, batch in enumerate(dataloader):
        batch = torch.cat([batch, batch[:, :, -2:], batch[:, :, -2:]], dim=-1)
        batch = batch[:, 0:1, :]
        optimizer.zero_grad()
        batch = batch.to(device)
        noise = add_noise(batch, noise_level).to(device)
        new_img = batch + noise
        pred_noise = model(noise, torch.Tensor(1).to(device))

        def gloss(a, b):
            return F.l1_loss(a, b)

        loss = gloss(noise, pred_noise)
        baseline = gloss(noise, noise * 0)
        loss.backward()
        optimizer.step()
        print(epoch, loss.item(), baseline.item(), (loss / baseline).item())
    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'model_weights')
        print('saved')

