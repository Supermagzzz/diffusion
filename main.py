import torch
from torch import nn
from model.model import SimpleDenoiser
from common import Common
import imageio.v2 as imageio
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

common = Common()

model = SimpleDenoiser(common)
model = nn.DataParallel(model)
model.to(common.device)
model.load_state_dict(torch.load('model_weights'))
print(common.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def print_example(data, index):
    plt.figure(figsize=(15, 3))
    for i in range(len(data)):
        path = 'tmp/' + str(i) + '.png'
        common.make_svg(data[i]).save_png(path)
        im = imageio.imread(path)
        plt.subplot(1, len(data), i + 1)
        plt.imshow(im)
    plt.savefig('trash/plt' + str(index))


for epoch in range(10000000):
    for step, batch in enumerate(common.dataloader):
        real = common.make_sample(batch)
        optimizer.zero_grad()

        t = torch.randint(0, common.T, (real.shape[0],), device=common.device).long()
        noised, noise = common.forward_diffusion_sample(real, t)

        pred_noise = model(noised, t)

        loss = common.calc_loss(noise, pred_noise)
        baseline = common.calc_loss(noise, -real * common.know_level)

        loss.backward()
        optimizer.step()
        print(epoch, loss.item(), baseline.item(), (loss / baseline).item())

    if epoch % 100 == 0:
        img = torch.randn((1, common.N, common.M_REAL), device=common.device)
        step = common.T // 10
        data = []
        for i in range(common.T - 1, -1, -1):
            t = torch.full((1,), i, device=common.device, dtype=torch.long)
            img = common.sample_timestep(img, t, model(img, t))
            if i % step == 0:
                data.append(img[0])
        print_example(data, epoch // 100)
        torch.save(model.state_dict(), 'model_weights')
        print('saved')

