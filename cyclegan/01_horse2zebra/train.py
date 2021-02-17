import argparse
import itertools
import os
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from models import ResNetGenerator, Discriminator
from datasets import UnalignedDataset
from losses import GANLoss
from utils import ImagePool, save_network


def args_initialize():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_size', type=int, default=286, help="")
    parser.add_argument('--fine_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--cuda', action='store_true')

    # lr, beta1
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning Rate")
    parser.add_argument('--beta1', type=float, default=0.5, help="Adams hyper parameter beta1")
    # save_epoch_freq, log_dir
    parser.add_argument('--save_freq', type=int, default=5, help="Epoch frequency of save timing")

    args = parser.parse_args()

    return args


def main():

    args = args_initialize()

    save_freq = args.save_freq
    epochs = args.num_epoch
    cuda = args.cuda

    train_dataset = UnalignedDataset(is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    net_G_A = ResNetGenerator(input_nc=3, output_nc=3)
    net_G_B = ResNetGenerator(input_nc=3, output_nc=3)
    net_D_A = Discriminator()
    net_D_B = Discriminator()

    if args.cuda:
        net_G_A = net_G_A.cuda()
        net_G_B = net_G_B.cuda()
        net_D_A = net_D_A.cuda()
        net_D_B = net_D_B.cuda()

    fake_A_pool = ImagePool(50)
    fake_B_pool = ImagePool(50)

    criterionGAN = GANLoss(cuda=cuda)
    criterionCycle = torch.nn.L1Loss()
    criterionIdt = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(
        itertools.chain(net_G_A.parameters(), net_G_B.parameters()),
        lr=args.lr,
        betas=(args.beta1, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(net_D_A.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D_B = torch.optim.Adam(net_D_B.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    log_dir = './logs'
    checkpoints_dir = './checkpoints'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    for epoch in range(epochs):

        running_loss = np.zeros((8))
        for batch_idx, data in enumerate(train_loader):

            input_A = data['A']
            input_B = data['B']

            if cuda:
                input_A = input_A.cuda()
                input_B = input_B.cuda()

            real_A = Variable(input_A)
            real_B = Variable(input_B)


            """
            Backward net_G
            """
            optimizer_G.zero_grad()
            lambda_idt = 0.5
            lambda_A = 10.0
            lambda_B = 10.0

            # 各 Generatorに変換後の画像を入力
            # 何もしないのが理想の出力
            idt_B = net_G_A(real_B)
            loss_idt_A = criterionIdt(idt_B, real_B) * lambda_B * lambda_idt

            idt_A = net_G_B(real_A)
            loss_idt_B = criterionIdt(idt_A, real_A) * lambda_A * lambda_idt

            # GAN loss = D_A(G_A(A))
            # G_Aとしては生成した偽物画像が本物(True)と判断して欲しい
            fake_B = net_G_A(real_A)
            pred_fake = net_D_A(fake_B)
            loss_G_A = criterionGAN(pred_fake, True)

            fake_A = net_G_B(real_B)
            pred_fake = net_D_B(fake_A)
            loss_G_B = criterionGAN(pred_fake, True)

            rec_A = net_G_B(fake_B)
            loss_cycle_A = criterionCycle(rec_A, real_A) * lambda_A

            rec_B = net_G_A(fake_A)
            loss_cycle_B = criterionCycle(rec_B, real_B) * lambda_B

            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            loss_G.backward()

            optimizer_G.step()

            """
            update D_A
            """
            optimizer_D_A.zero_grad()
            fake_B = fake_B_pool.query(fake_B.data)

            pred_real = net_D_A(real_B)
            loss_D_real = criterionGAN(pred_real, True)

            pred_fake = net_D_A(fake_B.detach())
            loss_D_fake = criterionGAN(pred_fake, False)

            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()

            """
            update D_B
            """
            optimizer_D_B.zero_grad()
            fake_A = fake_A_pool.query(fake_A.data)

            pred_real = net_D_B(real_A)
            loss_D_real = criterionGAN(pred_real, True)

            pred_fake = net_D_B(fake_A.detach())
            loss_D_fake = criterionGAN(pred_fake, False)

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()


            optimizer_D_B.step()

            ret_loss = np.array([
                loss_G_A.data.detach().cpu().numpy(), loss_D_A.data.detach().cpu().numpy(),
                loss_G_B.data.detach().cpu().numpy(), loss_D_B.data.detach().cpu().numpy(),
                loss_cycle_A.data.detach().cpu().numpy(), loss_cycle_B.data.detach().cpu().numpy(),
                loss_idt_A.data.detach().cpu().numpy(), loss_idt_B.data.detach().cpu().numpy()
            ])
            running_loss += ret_loss

            """
            Save checkpoints
            """
            if (epoch + 1) % save_freq == 0:
                save_network(net_G_A, 'G_A', str(epoch + 1))
                save_network(net_D_A, 'D_A', str(epoch + 1))
                save_network(net_G_B, 'G_B', str(epoch + 1))
                save_network(net_D_B, 'D_B', str(epoch + 1))

        running_loss /= len(train_loader)
        losses = running_loss
        print('epoch %d, losses: %s' % (epoch + 1, running_loss))

        writer.add_scalar('loss_G_A', losses[0], epoch)
        writer.add_scalar('loss_D_A', losses[1], epoch)
        writer.add_scalar('loss_G_B', losses[2], epoch)
        writer.add_scalar('loss_D_B', losses[3], epoch)
        writer.add_scalar('loss_cycle_A', losses[4], epoch)
        writer.add_scalar('loss_cycle_B', losses[5], epoch)
        writer.add_scalar('loss_idt_A', losses[6], epoch)
        writer.add_scalar('loss_idt_B', losses[7], epoch)


if __name__ == '__main__':
    main()
