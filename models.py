import torch
import torch.nn as nn


def conv_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )


def deconv_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_size=28):
        super(Discriminator, self).__init__()
        self.pipe = nn.Sequential(
            # 32*32*64
            conv_block(in_channels=1+1, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),

            # 16*16*128
            conv_block(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),

            # 8*8*256
            conv_block(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),

            # 4*4*512
            conv_block(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),

            # 1*1
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid()
        )
        self.embed = nn.Embedding(num_classes, img_size * img_size)
        self.img_size = img_size

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.pipe(x)


class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes=10):
        super(Generator, self).__init__()
        self.pipe = nn.Sequential(
            # 4*4*1024
            deconv_block(in_channels=noise_dim * 2, out_channels=512, kernel_size=(4, 4),
                         stride=(1, 1), padding=(0, 0)),
            # 8*8*256
            deconv_block(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),

            # 16*16*128
            deconv_block(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),

            # 32*32*64
            deconv_block(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),

            # 64*64
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(4, 4),
                               stride=(2, 2), padding=(1, 1), bias=False),
            nn.Tanh()
        )
        self.embed = nn.Embedding(num_classes, noise_dim)

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.pipe(x)


"""
if __name__ == '__main__':
    val_disc = torch.randn(2, 1, 28, 28)
    model_disc = Discriminator()
    print(f'disc: {model_disc(val_disc).shape}')

    noise_dim = 100
    val_gen = torch.randn(2, noise_dim, 1, 1)
    model_gen = Generator(noise_dim)
    print(f'gen: {model_gen(val_gen).shape}')
"""