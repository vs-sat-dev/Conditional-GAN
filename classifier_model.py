import torch
import torch.nn as nn


def conv_block(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, stride=stride,
                  padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # 64*64*32
        self.conv1 = conv_block(in_channels=1, out_channels=32)
        self.residual1 = nn.Sequential(conv_block(32, 32), conv_block(32, 32))

        # 32*32*64
        self.conv2 = conv_block(in_channels=32, out_channels=64, kernel_size=(4, 4),
                                stride=(2, 2), padding=(1, 1))
        self.residual2 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

        # 16*16*128
        self.conv3 = conv_block(in_channels=64, out_channels=128, kernel_size=(4, 4),
                                stride=(2, 2), padding=(1, 1))
        self.residual3 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        # 8*8*256
        self.conv4 = conv_block(in_channels=128, out_channels=256, kernel_size=(4, 4),
                                stride=(2, 2), padding=(1, 1))
        self.residual4 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))

        # 4*4*512
        self.conv5 = conv_block(in_channels=256, out_channels=512, kernel_size=(4, 4),
                                stride=(2, 2), padding=(1, 1))
        self.residual5 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        # 1*1*1024
        self.conv6 = conv_block(in_channels=512, out_channels=1024, kernel_size=(4, 4),
                                stride=(1, 1), padding=(0, 0))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(1024, 10))

    def forward(self, x):
        out = self.conv1(x)
        out = self.residual1(out) + out

        out = self.conv2(out)
        out = self.residual2(out) + out

        out = self.conv3(out)
        out = self.residual3(out) + out

        out = self.conv4(out)
        out = self.residual4(out) + out

        out = self.conv5(out)
        out = self.residual5(out) + out

        out = self.conv6(out)

        out = self.fc(out)

        return out

