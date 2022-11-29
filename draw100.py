import torch
from torch import nn
from tqdm import trange

from common import Common
import imageio.v2 as imageio
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

common = Common("", True)
# model = SimpleDenoiser(common)
# model = nn.DataParallel(model)
# model.to(common.device)
# model.load_state_dict(torch.load('model_weights'), strict=False)
# model.eval()
print(common.device)

for x in [0, 4, 7, 10, 12, 20]:
    common = Common("", True)
    def print_example(data, x):
        plt.figure(figsize=(40, 40))
        plt.figure(dpi=1200)
        fig, ax = plt.subplots(10, 10, figsize=(40, 40))
        for i in range(len(data)):
            path = 'tmp/100draw' + str(i) + '.png'
            common.make_svg(data[i]).save_png(path)
            im = imageio.imread(path)
            ax[i // 10, i % 10].imshow(im)
            ax[i // 10, i % 10].axis('off')
        plt.savefig('tmp/100_' + str(x) + '_')


    data = []
    for step, batch in enumerate(common.dataloader):
        if step == 100:
            break
        im = common.make_sample(batch)[0]
        sign = torch.randint(-1, 1, im.shape)
        noise_level = 8 / (2 ** x)
        im += sign * noise_level
        data.append(im)
    print_example(data, x)
