import numpy as np
import torch
from torch import nn

from dataset.dataloader import CustomImageDataset
from deepsvg.svglib.svg import SVG
from model import SimpleDenoiser

dataset = CustomImageDataset('../data/tensors')

def add_noise(tensor, mult):
    noise = []
    new_img = []
    for tens in tensor:
        noise.append([])
        new_img.append([])
        for row in tens:
            noise[-1].append([])
            new_img[-1].append([])
            for el in row:
                noi = np.random.normal() * mult
                noise[-1][-1].append(noi)
                new_img[-1][-1].append(el + noi)
    return torch.Tensor(new_img), torch.Tensor(noise)

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


model = SimpleDenoiser()
model.load_state_dict(torch.load('../model_weights'))
model.eval()
for i, img in enumerate(dataset):
    if i == 3:
        break
    new_img, noise = add_noise([img], 0.04)
    pred_noise = model(img, torch.Tensor(1))
    make_svg(new_img[0] - pred_noise[0]).save_svg('../trash/' + 'test' + str(i) + '.svg')
    make_svg(img).save_svg('../trash/' + 'real' + str(i) + '.svg')
    make_svg(new_img[0]).save_svg('../trash/' + 'inp' + str(i) + '.svg')
