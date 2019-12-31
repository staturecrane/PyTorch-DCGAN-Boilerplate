import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from src.models import Discriminator, Generator
from src.utils import truncated_noise_sample, weights_init


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="cifar10 | lsun | mnist |imagenet | folder | lfw | fake",
)
parser.add_argument("--dataroot", required=True, help="path to dataset")
parser.add_argument(
    "--workers", type=int, help="number of data loading workers", default=2
)
parser.add_argument("--batchSize", type=int, default=64, help="input batch size")
parser.add_argument(
    "--imageSize",
    type=int,
    default=64,
    help="the height / width of the input image to network",
)
parser.add_argument(
    "--nz", type=int, default=128, help="size of the la        tent z vector"
)
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--ndf", type=int, default=64)
parser.add_argument(
    "--niter", type=int, default=25, help="number of epochs to train for"
)
parser.add_argument(
    "--lr_g", type=float, default=0.0002, help="learning rate, default=0.0002"
)
parser.add_argument(
    "--lr_d", type=float, default=0.0002, help="learning rate, default=0.0002"
)
parser.add_argument(
    "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
)
parser.add_argument("--cuda", action="store_true", help="enables cuda")
parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
parser.add_argument(
    "--generator", default="", help="path to generator (to continue training)"
)
parser.add_argument(
    "--discriminator", default="", help="path to discriminator (to continue training)"
)
parser.add_argument(
    "--outf", default=".", help="folder to output images and model checkpoints"
)
parser.add_argument("--manualSeed", type=int, help="manual seed")
parser.add_argument(
    "--classes",
    default="bedroom",
    help="comma separated list of classes for the lsun data set",
)

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ["imagenet", "folder", "lfw"]:
    # folder dataset
    dataset = dset.ImageFolder(
        root=opt.dataroot,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.imageSize),
                transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    nc = 3
elif opt.dataset == "lsun":
    classes = [c + "_train" for c in opt.classes.split(",")]
    dataset = dset.LSUN(
        root=opt.dataroot,
        classes=classes,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.imageSize),
                transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    nc = 3
elif opt.dataset == "cifar10":
    dataset = dset.CIFAR10(
        root=opt.dataroot,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    nc = 3

elif opt.dataset == "mnist":
    dataset = dset.MNIST(
        root=opt.dataroot,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    nc = 1

elif opt.dataset == "fake":
    dataset = dset.FakeData(
        image_size=(3, opt.imageSize, opt.imageSize), transform=transforms.ToTensor()
    )
    nc = 3

assert dataset
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers)
)

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


generator = Generator(nz, nc, ngf, opt.imageSize, ngpu).to(device)
generator.apply(weights_init)
if opt.generator != "":
    generator.load_state_dict(torch.load(opt.generator))
print(generator)

discriminator = Discriminator(nc, ndf, opt.imageSize, ngpu).to(device)
discriminator.apply(weights_init)
if opt.discriminator != "":
    discriminator.load_state_dict(torch.load(opt.discriminator))
print(discriminator)

# setup optimizer
optimizerD = optim.Adam(
    discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999)
)
optimizerG = optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

fixed_noise = (
    torch.from_numpy(truncated_noise_sample(batch_size=64, dim_z=nz, truncation=0.4))
    .view(64, nz, 1, 1)
    .to(device)
)
real_label = 0.9
fake_label = 0

criterion = nn.BCELoss()

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        discriminator.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = discriminator(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = generator(noise)
        label.fill_(fake_label)
        output = discriminator(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        if i % 2 == 0:
            label.fill_(real_label)  # fake labels are real for generator cost
            output = discriminator(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

        print(
            "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
            % (
                epoch,
                opt.niter,
                i,
                len(dataloader),
                errD.item(),
                errG.item(),
                D_x,
                D_G_z1,
                D_G_z2,
            )
        )
        if i % 100 == 0:
            vutils.save_image(
                real_cpu, "%s/real_samples.png" % opt.outf, normalize=True
            )
            fake = generator(fixed_noise)
            vutils.save_image(
                fake.detach(),
                "%s/fake_samples_epoch_%03d.png" % (opt.outf, epoch),
                normalize=True,
            )

    # do checkpointing
    torch.save(generator.state_dict(), "%s/netG_epoch_%d.pth" % (opt.outf, epoch))
    torch.save(discriminator.state_dict(), "%s/netD_epoch_%d.pth" % (opt.outf, epoch))

