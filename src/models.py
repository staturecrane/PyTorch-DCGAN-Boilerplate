import torch
import torch.nn as nn

from src.spectral_norm import SpectralNorm


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, image_size, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        if image_size == 64:
            conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
            conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
            conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
            conv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
            conv5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

            nn.init.orthogonal_(conv1.weight.data)
            nn.init.orthogonal_(conv2.weight.data)
            nn.init.orthogonal_(conv3.weight.data)
            nn.init.orthogonal_(conv4.weight.data)
            nn.init.orthogonal_(conv5.weight.data)

            self.main = nn.Sequential(
                SpectralNorm(conv1),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(0.2),
                SpectralNorm(conv2),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(0.2),
                SpectralNorm(conv3),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(0.2),
                SpectralNorm(conv4),
                nn.BatchNorm2d(ngf),
                nn.ReLU(0.2),
                SpectralNorm(conv5),
                nn.Tanh(),
            )

        elif image_size == 128:
            conv1 = nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False)
            conv2 = nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False)
            conv3 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
            conv4 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
            conv5 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
            conv6 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

            nn.init.orthogonal_(conv1.weight.data)
            nn.init.orthogonal_(conv2.weight.data)
            nn.init.orthogonal_(conv3.weight.data)
            nn.init.orthogonal_(conv4.weight.data)
            nn.init.orthogonal_(conv5.weight.data)
            nn.init.orthogonal_(conv6.weight.data)

            self.main = nn.Sequential(
                SpectralNorm(conv1),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(0.2),
                SpectralNorm(conv2),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(0.2),
                SpectralNorm(conv3),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(0.2),
                SpectralNorm(conv4),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(0.2),
                SpectralNorm(conv5),
                nn.BatchNorm2d(ngf),
                nn.ReLU(0.2),
                SpectralNorm(conv6),
                nn.Tanh(),
            )

        elif image_size == 256:
            conv1 = nn.ConvTranspose2d(nz, ngf * 32, 4, 1, 0, bias=False)
            conv2 = nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False)
            conv3 = nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False)
            conv4 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
            conv5 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
            conv6 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
            conv7 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

            nn.init.orthogonal_(conv1.weight.data)
            nn.init.orthogonal_(conv2.weight.data)
            nn.init.orthogonal_(conv3.weight.data)
            nn.init.orthogonal_(conv4.weight.data)
            nn.init.orthogonal_(conv5.weight.data)
            nn.init.orthogonal_(conv6.weight.data)
            nn.init.orthogonal_(conv7.weight.data)

            self.main = nn.Sequential(
                SpectralNorm(conv1),
                nn.BatchNorm2d(ngf * 32),
                nn.ReLU(0.2),
                SpectralNorm(conv2),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(0.2),
                SpectralNorm(conv3),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(0.2),
                SpectralNorm(conv4),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(0.2),
                SpectralNorm(conv5),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(0.2),
                SpectralNorm(conv6),
                nn.BatchNorm2d(ngf),
                nn.ReLU(0.2),
                SpectralNorm(conv7),
                nn.Tanh(),
            )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc, ndf, image_size, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        if image_size == 64:
            conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
            conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
            conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
            conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
            conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

            nn.init.orthogonal_(conv1.weight.data)
            nn.init.orthogonal_(conv2.weight.data)
            nn.init.orthogonal_(conv3.weight.data)
            nn.init.orthogonal_(conv4.weight.data)
            nn.init.orthogonal_(conv5.weight.data)

            self.main = nn.Sequential(
                SpectralNorm(conv1),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv2),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv3),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv4),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv5),
                nn.Sigmoid(),
            )
        elif image_size == 128:
            conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
            conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
            conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
            conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
            conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)
            conv6 = nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)

            nn.init.orthogonal_(conv1.weight.data)
            nn.init.orthogonal_(conv2.weight.data)
            nn.init.orthogonal_(conv3.weight.data)
            nn.init.orthogonal_(conv4.weight.data)
            nn.init.orthogonal_(conv5.weight.data)
            nn.init.orthogonal_(conv6.weight.data)

            self.main = nn.Sequential(
                SpectralNorm(conv1),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv2),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv3),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv4),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv5),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv6),
                nn.Sigmoid(),
            )
        elif image_size == 256:
            conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
            conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
            conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
            conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
            conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)
            conv6 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False)
            conv7 = nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False)

            nn.init.orthogonal_(conv1.weight.data)
            nn.init.orthogonal_(conv2.weight.data)
            nn.init.orthogonal_(conv3.weight.data)
            nn.init.orthogonal_(conv4.weight.data)
            nn.init.orthogonal_(conv5.weight.data)
            nn.init.orthogonal_(conv6.weight.data)
            nn.init.orthogonal_(conv7.weight.data)

            self.main = nn.Sequential(
                SpectralNorm(conv1),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv2),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv3),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv4),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv5),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv6),
                nn.BatchNorm2d(ndf * 32),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(conv7),
                nn.Sigmoid(),
            )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

