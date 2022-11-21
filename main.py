import math
import random

import torch
from torch import nn
from torch.nn import BCELoss

from model.model import SimpleDenoiser
from model.discriminator import SimpleDiscriminator
from common import Common
import imageio.v2 as imageio
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

common = Common()
autoencoder = SimpleDenoiser(common)
autoencoder = nn.DataParallel(autoencoder)
autoencoder.to(common.device)
#autoencoder.load_state_dict(torch.load('autoencoder_weights'), strict=False)

discriminator = SimpleDiscriminator(common)
discriminator = nn.DataParallel(discriminator)
discriminator.to(common.device)
#discriminator.load_state_dict(torch.load('discriminator_weights'), strict=False)

print(common.device)
optimizer_encoder = torch.optim.AdamW(autoencoder.parameters(), lr=0.00001)
optimizer_discriminator = torch.optim.AdamW(discriminator.parameters(), lr=0.00001)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

bce = BCELoss()

all_losses = []
baseline = 0
for epoch in range(10000000):
    for step, batch in enumerate(common.dataloader):
        batch_size = batch.shape[0]
        real = common.make_sample(batch)

        # train discriminator
        optimizer_encoder.zero_grad()
        optimizer_discriminator.zero_grad()

        pred = autoencoder(real)

        real_prob = sigmoid(discriminator(real))
        fake_prob = sigmoid(discriminator(pred))

        real_label = torch.ones((batch_size,), dtype=torch.float, device=common.device)
        fake_label = torch.zeros((batch_size,), dtype=torch.float, device=common.device)

        loss_discriminator = bce(real_prob, real_label) + bce(fake_prob, fake_label)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # train autoencoder
        optimizer_encoder.zero_grad()
        optimizer_discriminator.zero_grad()

        pred = autoencoder(real)

        fake_prob = sigmoid(discriminator(pred))
        real_label = torch.ones((batch_size,), dtype=torch.float, device=common.device)
        loss_mse = -torch.log(common.calc_loss(pred, real))
        loss_gan = bce(fake_prob, real_label)
        loss_autoencoder = loss_mse + loss_gan
        loss_autoencoder.backward()
        optimizer_encoder.step()

        all_losses.append(torch.log(loss_autoencoder))
        print(epoch, loss_autoencoder.item(), loss_discriminator.item(), loss_mse.item(), loss_gan.item())

    if epoch % 100 == 0:
        torch.save(autoencoder.state_dict(), 'autoencoder_weights')
        torch.save(discriminator.state_dict(), 'discriminator_weights')
        print('saved')
