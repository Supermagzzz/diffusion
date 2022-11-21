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

autoencoder_loss = []
discriminator_loss = []
mse_loss = []
gan_loss = []
baseline = 0
for epoch in range(10000000):
    for step, batch in enumerate(common.dataloader):
        batch_size = batch.shape[0]
        real = common.make_sample(batch)

        pred = autoencoder(real)
        real_label = torch.ones((batch_size,), dtype=torch.float, device=common.device)
        fake_label = torch.zeros((batch_size,), dtype=torch.float, device=common.device)

        # train discriminator
        if len(discriminator_loss) == 0 or gan_loss[-1] < 3:
            discriminator.zero_grad()
            real_prob = sigmoid(discriminator(real))
            fake_prob = sigmoid(discriminator(pred))
            loss_discriminator = bce(real_prob, real_label) + bce(fake_prob, fake_label)
            loss_discriminator.backward()
            optimizer_discriminator.step()
            discriminator_loss.append(loss_discriminator.item())
        else:
            discriminator_loss.append(discriminator[-1])

        # train autoencoder
        autoencoder.zero_grad()
        fake_prob = sigmoid(discriminator(pred))
        loss_mse = common.calc_loss(pred, real) * 50
        loss_gan = bce(fake_prob, real_label)
        loss_autoencoder = loss_mse + loss_gan
        loss_autoencoder.backward()
        optimizer_encoder.step()

        autoencoder_loss.append(loss_autoencoder.item())
        mse_loss.append(loss_mse.item())
        gan_loss.append(loss_gan.item())

        print(epoch, autoencoder_loss[-1], discriminator_loss[-1], mse_loss[-1], gan_loss[-1])

    if epoch % 100 == 0:
        torch.save(autoencoder.state_dict(), 'autoencoder_weights')
        torch.save(discriminator.state_dict(), 'discriminator_weights')
        plt.plot(autoencoder_loss, label='autoencoder_loss')
        plt.plot(discriminator_loss, label='discriminator_loss')
        plt.plot(mse_loss, label='mse_loss')
        plt.plot(gan_loss, label='gan_loss')
        plt.legend()
        plt.savefig('trash/gan' + str(epoch // 100) + '.png')
        plt.show()
        print('saved')
