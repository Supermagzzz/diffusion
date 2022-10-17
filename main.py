import torch
import numpy as np
from torch.optim import Adam
from torch.nn.functional import l1_loss

from dataset.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
from model.model import SimpleDenoiser

dataset = CustomImageDataset('data/tensors')
dataloader = DataLoader(dataset, batch_size=len(dataset.images), shuffle=False, drop_last=True)


def add_noise(tensor, mult):
    return torch.normal(0, 1, size=tensor.shape) * mult


model = SimpleDenoiser()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
new_img, noise = None, None
for epoch in range(1000):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        noise = add_noise(batch, 0.001)
        new_img = batch + noise
        loss = l1_loss(model(new_img, torch.Tensor(1)), noise)
        baseline = l1_loss(noise * 0, noise)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(epoch, loss.item(), baseline.item(), loss.item() - baseline.item())
    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'model_weights')
        print('saved')

