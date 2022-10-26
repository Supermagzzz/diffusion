import torch

from dataset.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from deepsvg.svglib.svg import SVG


class Common:
    def __init__(self, check=False):
        self.N = 5
        self.M = 20
        self.HIDDEN = 256
        self.BLOCKS = 2000000
        self.M_REAL = self.M * 6 + 6
        self.noise_level = 0.03
        self.know_level = 0.01
        self.batch_sz = 1
        self.cpu_batch_sz = 1
        self.apply_batch_sz = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = CustomImageDataset('data/tensors')

        if check:
            self.real_batch_sz = self.apply_batch_sz
        else:
            self.real_batch_sz = self.cpu_batch_sz if self.device == "cpu" else self.batch_sz

        self.dataloader = DataLoader(self.dataset, self.real_batch_sz, shuffle=False, drop_last=True)

    def calc_loss(self, a, b):
        return F.l1_loss(a, b)
        return (a - b).pow(2).sum()

    def make_sample(self, batch):
        batch = torch.cat([batch[:, :, :2], batch[:, :, :2], batch], dim=-1)
        return batch.to(self.device)

    def add_noise(self, tensor, mult):
        return (torch.normal(0, mult, size=tensor.shape).to(self.device) - tensor * self.know_level).to(self.device)

    def make_svg(self, tensor):
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
        return SVG.from_tensor(torch.Tensor(data))

