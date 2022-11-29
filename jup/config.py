import hashlib

import matplotlib.pyplot as plt
import pydiffvg
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import pickle
import os
from transformers import BertConfig, BertModel
import math
from deepsvg.svglib.svg import SVG
import imageio
from torch.nn import BCELoss

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

SEQ_RANGE = 1
BATCH_SIZE = 256 if torch.cuda.is_available() else 2
N = 5
M = 8
M_REAL = M * 6 + 6
BLOCKS = 2 ** 13
HIDDEN = 256
lr = 1e-4
output_path = "."
input_path = "."

def load_dataset(img_dir):
    images = []
    for i in os.listdir(img_dir):
        if i.endswith('.pkl'):
            img_path = os.path.join(img_dir, i)
            data = pickle.load(open(img_path, 'rb'))
            images.append((img_path, data))
    images.sort(key=lambda path_and_im: path_and_im[0].encode())
    imgs = [im for path, im in images]
    return imgs

def make_svg(tensor):
    tensor += 0.7
    tensor *= 17
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
    svg = SVG.from_tensor(torch.Tensor(data))
    return svg


def make_png(svg):
    # with torch.no_grad():
    canvas_width, canvas_height = 50, 50
    el = (svg + 0.5) * int(0.8 * canvas_width) + int(0.1 * canvas_width)

    num_control_points = torch.ones((el.shape[1] // 6 - 1)) * 2
    points = torch.cat([el[:, :2], el[:, 6:]], dim=1)
    points = points.reshape(points.shape[0], points.shape[1] // 2, 2)
    pathes = [pydiffvg.Path(num_control_points, row, False) for row in points]
    groups = [pydiffvg.ShapeGroup(shape_ids=torch.tensor([0, 1, 2, 3, 4]), fill_color=None, stroke_color=torch.tensor([0, 0, 0, 1]))]

    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, pathes, groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width,  # width
                 canvas_height,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 1,  # seed
                 None,
                 *scene_args)
    # return img
    pydiffvg.imwrite(img.cpu(), 'test.png', gamma=2.2)

def make_png_batch(x):
    return torch.stack([make_png(svg) for svg in x])

def make_batch(batch_size, data):
    return torch.stack([data for x in range(batch_size)])


def make_seq(seq_size, data):
    data = data.reshape(data.shape[0], 1, data.shape[-1])
    return torch.cat([data for x in range(seq_size)], dim=1)

