import pydiffvg
import torch
import numpy as np
from torch.optim import Adam
from torch.nn.functional import l1_loss

from dataset.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
from model.model import SimpleDenoiser
import diffvg

dataset = CustomImageDataset('data/tensors')
dataloader = DataLoader(dataset, batch_size=len(dataset.images) // 2, shuffle=False, drop_last=True)


def add_noise(tensor, mult):
    return torch.normal(0, 1, size=tensor.shape) * mult


def make_image(el):
    return el
    canvas_width = 100
    canvas_height = 100
    data = (el - torch.min(el) + 0.5) * 50
    pathes = []
    groups = []
    for j, row in enumerate(data):
        num_control_points = []
        points = []
        points.append((row[0], row[1]))
        for i in range(2, row.shape[0], 6):
            num_control_points.append(2)
            points.append((row[i], row[i + 1]))
            points.append((row[i + 2], row[i + 3]))
            points.append((row[i + 4], row[i + 5]))
        pathes.append(pydiffvg.Path(torch.Tensor(num_control_points), torch.Tensor(points), False))
        groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([j]), fill_color=None,
                                          stroke_color=torch.tensor([0, 0, 0, 1.0])))
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, pathes, groups)
    render = pydiffvg.RenderFunction.apply
    img = render(256,  # width
                 256,  # height
                 5,  # num_samples_x
                 5,  # num_samples_y
                 0,  # seed
                 None,
                 *scene_args)
    return img


model = SimpleDenoiser()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
new_img, noise = None, None
for epoch in range(1000):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        noise = add_noise(batch, 0.01)
        new_img = batch + noise
        pred_noise = model(new_img.to(device), torch.Tensor(1).to(device))
        pred_batch = new_img - pred_noise
        losses = []
        for i in range(batch.shape[0]):
            losses.append((make_image(batch[i]) - make_image(pred_batch[i])).pow(2).sum())
        loss = sum(losses)
        baseline = (new_img - batch).pow(2).sum()
        loss.backward()
        optimizer.step()
        if step == 0:
            print(epoch, loss.item(), baseline.item(), loss.item() - baseline.item())
    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'model_weights')
        print('saved')

