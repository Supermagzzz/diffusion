from config import *
from model import *

train_dataset = load_dataset(input_path + '/vector_dataset')

m = len(train_dataset)

train_data, val_data = random_split(train_dataset, [m - int(m * 0.2), int(m * 0.2)])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, drop_last=True)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, drop_last=True)

def plot_ae_outputs(vae, epoch, n=10):
    plt.figure(figsize=(16, 4.5))

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = make_batch(1, train_data[i].to(device))
        vae.eval()
        vae.eval()
        with torch.no_grad():
            rec_img, _ = vae(img)

        path = 'tmp/100draw' + str(i) + '.png'
        make_svg(img[0]).save_png(path)
        im = imageio.imread(path)

        plt.imshow(im, cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n // 2:
            ax.set_title('Original images')

        path = 'tmp/100draw' + str(i) + '.png'
        make_svg(rec_img[0]).save_png(path)
        rec_im = imageio.imread(path)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_im, cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n // 2:
            ax.set_title('Reconstructed images')
    plt.savefig('trash/' + 'example_' + str(epoch))


bce = BCELoss()

def train_epoch(vae, dis, device, dataloader, optim_vae, optim_dis):
    vae.train()
    dis.train()
    train_loss_mse = 0.0
    train_loss_kl = 0.0
    train_loss_bce = 0.0
    train_loss_dis = 0.0
    pbar = dataloader
    steps = 0
    for x in pbar:
        steps += 1
        x = x.to(device)

        true_label = torch.ones((BATCH_SIZE,), dtype=torch.float, device=device)
        false_label = torch.zeros((BATCH_SIZE,), dtype=torch.float, device=device)

        # dis
        if True:
            x_fake, _ = vae(x)
            real_prob = dis(x)
            fake_prob = dis(x_fake)

            loss_dis = (bce(real_prob, true_label) + bce(fake_prob, false_label)) / 2 / 4

            optim_dis.zero_grad()
            loss_dis.backward()
            optim_dis.step()

        # vae
        if True:
            x_fake, vae_kl = vae(x)
            fake_prob = dis(x_fake)

            loss_mse = ((make_png_batch(x) - make_png_batch(x_fake)) ** 2).mean() * 100
            loss_kl = vae_kl.mean() / 100
            loss_bce = bce(fake_prob, true_label) / 4
            loss_vae = loss_mse + loss_kl + loss_bce

            optim_vae.zero_grad()
            loss_vae.backward()
            optim_vae.step()

        train_loss_mse += loss_mse.item()
        train_loss_bce += loss_bce.item()
        train_loss_kl += loss_kl.item()
        train_loss_dis += loss_dis.item()
        if device == torch.device("cpu"):
            break

    train_loss_mse /= steps
    train_loss_bce /= steps
    train_loss_kl /= steps
    train_loss_dis /= steps

    print("%d: train, var_mse: %0.3f, var_kl: %0.3f, var_bce: %0.3f, dis: %0.3f" % (
        epoch, train_loss_mse, train_loss_kl, train_loss_bce, train_loss_dis))


torch.manual_seed(0)

vae = nn.DataParallel(VariationalAutoencoder())
dis = nn.DataParallel(Discriminator())

optim_vae = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-5)
optim_dis = torch.optim.AdamW(dis.parameters(), lr=lr, weight_decay=1e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

vae.to(device)
dis.to(device)

if __name__ == "__main__":
    epoch = 0
    while True:
        train_epoch(vae, dis, device, train_loader, optim_vae, optim_dis)
        if epoch % 10 == 0:
            plot_ae_outputs(vae, epoch, n=10)
        if epoch % 100 == 0:
            torch.save(vae.state_dict(), output_path + '/vae')
            torch.save(dis.state_dict(), output_path + '/dis')
        epoch += 1
