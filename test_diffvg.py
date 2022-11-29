import torch
import numpy as np
from torch.optim import Adam
from torch.nn.functional import l1_loss

from dataset.dataloader import CustomImageDataset
from torch.utils.data import DataLoader

from deepsvg.svglib.svg import SVG
import diffvg
import pydiffvg

canvas_width = 100
canvas_height = 100


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
for j, el in enumerate(dataset):
    if j <= 4:
        continue
    canvas_width, canvas_height = 100, 100
    el += 0.5
    el *= int(0.8 * canvas_width)
    el += int(0.1 * canvas_width)
    pathes = []
    groups = []
    for j, row in enumerate(el):
        num_control_points = []
        points = []
        points.append((row[0], row[1]))
        for i in range(2, row.shape[0], 6):
            num_control_points.append(2)
            points.append((row[i + 2], row[i + 3]))
            points.append((row[i], row[i + 1]))
            points.append((row[i + 4], row[i + 5]))
        pathes.append(pydiffvg.Path(torch.Tensor(num_control_points), torch.Tensor(points), True))
    groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([0, 1, 2, 3, 4]), fill_color=None, stroke_color=torch.tensor([0, 0, 0, 1])))
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, pathes, groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width,  # width
                 canvas_height,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 1,  # seed
                 None,
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), 'test.png', gamma=2.2)
    break
