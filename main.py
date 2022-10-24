import torch

from dataset.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
from model.model import SimpleDenoiser

torch.set_default_dtype(torch.float32)
noise_level = 0.03
know_level = 0.01
batch_sz = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CustomImageDataset('data/tensors')
dataloader = DataLoader(dataset, batch_size=batch_sz if device == "cpu" else batch_sz, shuffle=False, drop_last=True)


def add_noise(tensor, mult):
    return torch.normal(0, mult, size=tensor.shape).to(device) - tensor * know_level


N = 5
M = 8

model = SimpleDenoiser(noise_level, device)
model.to(device)
print(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
new_img, noise = None, None
for epoch in range(100000):
    for step, batch in enumerate(dataloader):
        batch = torch.cat([batch[:, :, :2], batch[:, :, :2], batch], dim=-1)
        for i in range(batch.shape[0]):
            batch[i, :, :] = i
        optimizer.zero_grad()
        batch = batch.to(device)
        noise = add_noise(batch, noise_level).to(device)
        new_img = batch# + noise
        pred_noise = model(new_img, torch.Tensor(1).to(device))

        def gloss(a, b):
            return (a - b).pow(2).sum()

        loss = gloss(noise, pred_noise)
        baseline = gloss(noise, -batch * know_level)
        loss.backward()
        optimizer.step()
        print(epoch, loss.item(), baseline.item(), (loss / baseline).item())
    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'model_weights')
        print('saved')

