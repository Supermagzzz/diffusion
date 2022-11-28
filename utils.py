import torch

from dataset.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from deepsvg.svglib.svg import SVG


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
        startX, startY = row[0], row[1]
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
            if i + 6 == row.shape[0]:
                command[-2] = startX
                command[-1] = startY
            else:
                command[-2] = row[i + 4]
                command[-1] = row[i + 5]
            lastX, lastY = command[-2], command[-1]
            data.append(command)
    return SVG.from_tensor(torch.Tensor(data))
