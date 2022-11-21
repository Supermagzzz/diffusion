import torch
from torch import nn
from tqdm import trange

from model.model import SimpleDenoiser
from common import Common
import imageio.v2 as imageio
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

common = Common(check=True)
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
        path = 'tmp/100draw_with_real' + str(i) + '.png'
        common.make_svg(data[i]).save_png(path)
        im = imageio.imread(path)
        ax[i // 10, i % 10].imshow(im)
        ax[i // 10, i % 10].axis('off')
    plt.savefig('trash/100_with_real')


img = torch.randn((100, common.N, common.M_REAL), device=common.device)
step = common.T // 10
data = []
for i in trange(common.T - 1, -1, -1):
    for step, batch in enumerate(common.dataloader):
        real = common.make_sample(batch)
        data.append(real[0])
        pred = model(batch)
        data.append(pred)
print_example(data)
