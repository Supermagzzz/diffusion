import torch
import numpy as np
from torch.optim import Adam
from torch.nn.functional import l1_loss

from dataset.dataloader import CustomImageDataset
from torch.utils.data import DataLoader

from deepsvg.svglib.svg import SVG
from model.model import SimpleDenoiser
import diffvg
import pydiffvg

canvas_width = 256
canvas_height = 256


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
        for i in range(2, row.shape[0], 6):
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
            if (lastX - command[-2]) ** 2 + (lastY - command[-1]) ** 2 < 0.1:
                continue
            lastX, lastY = command[-2], command[-1]
            data.append(command)
    return SVG.from_tensor(torch.Tensor(data))

dataset = CustomImageDataset('data/tensors')
for el in dataset:
    el -= torch.min(el) - 0.5
    el *= 100
    pathes = []
    groups = []
    for j, row in enumerate(el):
        num_control_points = []
        points = []
        points.append((row[0], row[1]))
        for i in range(2, row.shape[0], 6):
            num_control_points.append(2)
            points.append((row[i], row[i + 1]))
            points.append((row[i + 2], row[i + 3]))
            points.append((row[i + 4], row[i + 5]))
        pathes.append(pydiffvg.Path(torch.Tensor(num_control_points), torch.Tensor(points), False))
        groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([j]), fill_color=None, stroke_color=torch.tensor([0, 0, 0, 1.0])))
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, pathes, groups)
    render = pydiffvg.RenderFunction.apply
    img = render(256,  # width
                 256,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 0,  # seed
                 None,
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), 'test.png', gamma=2.2)
    break

