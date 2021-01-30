import argparse
import random
import os

import h5py
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

manual_seed = 999

print("Random  Seed: ", manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)


class Generator(nn.Module):
    def __init__(self, ngpu=0, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu=0, ndf=64, ngf=64, nc=3):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def argparser():
    parser = argparse.ArgumentParser(description='DCGAN Training Script')

    parser.add_argument('--dataroot', type=str, default='/code/input/celeba-dataset/img_align_celeba', help="dir path of dataset")
    parser.add_argument('--output_dir', type=str, default='./output', help="output directory")

    parser.add_argument('--workers', type=int, default=4, help="Number of DataLoaders workers")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch Size")
    parser.add_argument('--image_size', type=int, default=64, help="Image Size Generator")
    parser.add_argument('--nc', type=int, default=3 , help="Color dimension")
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=5, help="Number of Epochs")
    parser.add_argument('--lr', type=float, default=0.0002, help="Learning Rate")
    parser.add_argument('--beta1', type=float, default=0.5, help="beta value for Adam optimizer")
    parser.add_argument('--ngpu', type=int, default=1, help="Number of GPU")

    args = parser.parse_args()

    return args


def main():

    args = argparser()
    HYPER_PARAMS = args.__dict__

    dataset = dset.ImageFolder(
        root=HYPER_PARAMS['dataroot'],
        transform=transforms.Compose([
            transforms.Resize(HYPER_PARAMS['image_size']),
            transforms.CenterCrop(HYPER_PARAMS['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5, 0.5)),
        ])
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=HYPER_PARAMS['batch_size'],
        shuffle=True,
        num_workers=HYPER_PARAMS['workers']
    )

    device = torch.device("cuda:0" if (torch.cuda.is_available() and HYPER_PARAMS['ngpu'] > 0) else "cpu")

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    """
    Create Generator
    """
    print('>>>>> Create Generator')
    netG = Generator(
        ngpu=HYPER_PARAMS['ngpu'],
        ngf=HYPER_PARAMS['ngf'],
        nc=HYPER_PARAMS['nc'],
        nz=HYPER_PARAMS['nz']
    ).to(device)

    netG.apply(weights_init)

    print(netG)

    """
    Create Discriminator
    """
    print('>>>>> Create Discriminator')

    netD = Discriminator(
        ngpu=HYPER_PARAMS['ngpu'],
        ngf=HYPER_PARAMS['ngf'],
        nc=HYPER_PARAMS['nc'],
        ndf=HYPER_PARAMS['ndf']
    ).to(device)

    netD.apply(weights_init)

    print(netD)

    """
    Loss Functions and Optimizers
    """
    print(">>>>> Loss Functions and Optimizers")

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(HYPER_PARAMS['batch_size'], HYPER_PARAMS['nz'], 1, 1, device=device)

    real_label = 1
    fake_label = 0

    optimizerG = optim.Adam(netG.parameters(), lr=HYPER_PARAMS['lr'], betas=(HYPER_PARAMS['beta1'], 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=HYPER_PARAMS['lr'], betas=(HYPER_PARAMS['beta1'], 0.999))


    """
    Training Loop
    """

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print(">>>>> Starting Training Loop...")

    for epoch in range(HYPER_PARAMS['num_epochs']):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)

            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, HYPER_PARAMS['nz'], 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake

            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake).view(-1)

            errG = criterion(output, label)

            errG.backward()

            D_G_z2 = output.mean().item()

            optimizerG.step()

            if i % 50 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, HYPER_PARAMS['num_epochs'], i,
                       len(dataloader),errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
                )

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == HYPER_PARAMS['num_epochs'] - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    os.makedirs(HYPER_PARAMS['output_dir'], exist_ok=True)

    hf_1 = h5py.File(os.path.join(HYPER_PARAMS['output_dir'], 'logs.h5'), 'w')
    hf_1.create_dataset('G_losses', data=G_losses)
    hf_1.create_dataset('D_losses', data=D_losses)

    torch.save(img_list, os.path.join(HYPER_PARAMS['output_dir'], 'gen_image.pth'))
    torch.save(netG.state_dict(), os.path.join(HYPER_PARAMS['output_dir'], 'Generator.pth'))
    torch.save(netD.state_dict(), os.path.join(HYPER_PARAMS['output_dir'], 'Discriminator.pth'))


if __name__ == '__main__':
    main()
