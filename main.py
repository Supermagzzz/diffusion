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
#model.load_state_dict(torch.load('model_weights'), strict=False)
print(common.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

all_losses = []
baseline = 0
for epoch in range(10000000):
    for step, batch in enumerate(common.dataloader):
        real = common.make_sample(batch)
        optimizer.zero_grad()

        pred = model(real)

        loss = common.calc_loss(pred, real)

        all_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        print(epoch, loss.item())

    if epoch % 100 == 0:
        torch.save(model.state_dict(), 'model_weights')
        print('saved')
