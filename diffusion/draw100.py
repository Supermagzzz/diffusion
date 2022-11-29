import torch
from torch import nn
from tqdm import trange

from model.model import SimpleDenoiser
from common import Common
import imageio.v2 as imageio
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

common = Common()
model = SimpleDenoiser(common)
model = nn.DataParallel(model)
model.to(common.device)
model.load_state_dict(torch.load('model_weights'), strict=False)
model.eval()
print(common.device)


def print_example(data):
    plt.figure(figsize=(40, 40))
    plt.figure(dpi=1200)
    fig, ax = plt.subplots(10, 10, figsize=(40, 40))
    for i in range(len(data)):
        path = 'tmp/100draw' + str(i) + '.png'
        common.make_svg(data[i]).save_png(path)
        im = imageio.imread(path)
        ax[i // 10, i % 10].imshow(im)
        ax[i // 10, i % 10].axis('off')
    plt.savefig('tmp/100')


img = torch.randn((100, common.N, common.M_REAL), device=common.device)
step = common.T // 10
data = []
for i in trange(common.T - 1, -1, -1):
    t = torch.full((100,), i, device=common.device, dtype=torch.long)
    img = common.sample_timestep(img, t, model(img, t))
for im in img:
    data.append(im)
print_example(data)
