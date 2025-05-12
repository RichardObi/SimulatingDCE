import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
            DoubleConv(in_channels, out_channels, kernel_size, padding, dropout)
        )

    def forward(self, x):
        return self.down_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, padding, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff1 = x2.size()[2] - x1.size()[2]
        diff2 = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff2 // 2, diff2 - diff2 // 2,
                        diff1 // 2, diff1 - diff1 // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, conv_size=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 2 x [(1x1) Conv + BN + LReLU]
        self.inc = (DoubleConv(in_channels=n_channels, out_channels=conv_size, kernel_size=1, padding=0))
        # downsample + 2 x [(3x3) Conv + BN + LReLU]
        self.down1 = (Down(in_channels=conv_size, out_channels=conv_size*2))
        # downsample + 2 x [(3x3) Conv + BN + Dropout + LReLU]
        self.down2 = (Down(in_channels=conv_size*2, out_channels=conv_size*4, dropout=0.5))
        # downsample + 2 x [(3x3) Conv + BN + Dropout + LReLU]
        self.down3 = (Down(in_channels=conv_size*4, out_channels=conv_size*8, dropout=0.5))
        # upsample + 2 x [(3x3) Conv + BN + Dropout + LReLU]
        self.up1 = (Up(in_channels=conv_size*8, out_channels=conv_size*4, dropout=0.5))
        # upsample + 2 x[(3x3) Conv + BN + LReLU]
        self.up2 = (Up(conv_size*4, conv_size*2))
        # upsample + 2 x[(3x3) Conv + BN + LReLU]
        self.up3 = (Up(conv_size*2, conv_size))
        # (1x1) Conv
        self.outc = (OutConv(conv_size, n_classes))
        # sigmoid activation
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, -1)      # channels first

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)

        shape = (logits.shape[0], self.n_classes, *logits.shape[2:4], self.n_channels)
        logits = torch.reshape(logits, shape)
        out = self.act(logits)

        return out
