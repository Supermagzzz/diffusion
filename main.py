import math
import sys
sys.path.append('/Users/maxim-kuzin/ml/pythonProject/diffvg')

import pydiffvg
import torch
import numpy as np
from torch.optim import Adam
from torch.nn.functional import l1_loss

from dataset.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
from model.model import SimpleDenoiser

dataset = CustomImageDataset('data/tensors')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)


def add_noise(tensor, mult):
    return torch.normal(0, 1, size=tensor.shape) * mult

N = 5
M = 20

def make_image(inp):
    # return el
    canvas_width, canvas_height = 64, 64
    el = (inp + 0.5)
    el *= int(0.8 * canvas_width)
    el += int(0.1 * canvas_width)

    all_ims = []
    for j, row in enumerate(el):
        for i in range(2, row.shape[0], 6):
            pathes = []
            groups = []
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
            pathes.append(pydiffvg.Path(torch.Tensor(num_control_points), points, False))
            groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(groups)]), fill_color=None,
                                              stroke_color=torch.Tensor([0, 0, 0, 1])))
            scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, pathes, groups)
            render = pydiffvg.RenderFunction.apply
            img = render(canvas_width,  # width
                         canvas_height,  # height
                         2,  # num_samples_x
                         2,  # num_samples_y
                         1,  # seed
                         None,
                         *scene_args)
            all_ims.append(img)
    return torch.cat(all_ims, dim=-1)


torch.autograd.set_detect_anomaly(True)
model = SimpleDenoiser()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
new_img, noise = None, None
for epoch in range(100000):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        noise = add_noise(batch, 0.01).to(device)
        new_img = batch + noise
        png = torch.zeros((batch.shape[0], 64, 64, 4 * N * M)).to(device)
        for i in range(batch.shape[0]):
            png[i] = make_image(new_img[i])
        pred_noise = model(png, new_img, torch.Tensor(1).to(device))
        loss = (noise - pred_noise).pow(2).sum()
        baseline = (noise - noise * 0).pow(2).sum()
        loss.backward()
        optimizer.step()
        print(epoch, loss.item(), baseline.item(), loss.item() - baseline.item())
    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'model_weights')
        print('saved')

