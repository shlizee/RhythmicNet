"""
RhyhmicNet stage 2 Rhythm2Drum (generating drum style (velocity and offsets)) code of "How does it sound?: Generation of Rhythmic
Soundtracks for Human Movement Videos" 
Copyright (c) 2021-2022 University of Washington. Developed in UW NeuroAI Lab by Xiulong Liu.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        model = [nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False)]
        if normalize:
            model.append(nn.BatchNorm2d(out_size))
        model.append(nn.LeakyReLU(0.2))
        if dropout:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        model = [
            nn.ConvTranspose2d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x, skip_input):
        x = self.model(x)
        out = torch.cat((x, skip_input), 1)
        return out


class Onset2VO(nn.Module):
    def __init__(self, input_shape):
        super(Onset2VO, self).__init__()
        channels, _, _ = input_shape
        self.down1 = UNetDown(channels, 16, normalize=False)
        self.down2 = UNetDown(16, 32)
        self.down3 = UNetDown(32, 64, dropout=0.15)
        self.down4 = UNetDown(64, 128, dropout=0.15)
        self.down5 = UNetDown(128, 128, dropout=0.15)
        # self.down5 = UNetDown(128, 256, dropout=0.15)
        # self.down6 = UNetDown(256, 256, dropout=0.15)

        # self.up1 = UNetUp(256, 128, dropout=0.15)
        # self.up2 = UNetUp(256+128, 64, dropout=0.15)
        self.up1 = UNetUp(128, 64, dropout=0.15)
        self.up2 = UNetUp(128+64, 32)
        self.up3 = UNetUp(64+32, 16)
        self.up4 = UNetUp(32+16, 4)
        self.conv1d = nn.Conv2d(20, 2, kernel_size=1)
        self.v_loss = nn.MSELoss()
        self.o_loss = nn.MSELoss()


    def forward(self, x, target=None):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        # d5 = self.down5(d4)
        # d6 = self.down6(d5) + noise[:,:, None, None]
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        # u5 = self.up5(u4, d1)
        out = self.conv1d(u4)
        velocity = torch.sigmoid(out[:,0,:,:])
        offsets = torch.tanh(out[:,1,:,:])
        if target is not None:
            v_loss = self.v_loss(velocity, target[:,0,:,:])
            o_loss = self.o_loss(offsets, target[:,1,:,:])
            loss = v_loss + o_loss
            return velocity, offsets, loss,  v_loss, o_loss
        return velocity, offsets

if __name__ == "__main__":
    input_shape = (1, 9, 32)
    gnet = Onset2VO(input_shape)
    gnet.cuda()
    imgs = torch.rand((6,1,18,64)).cuda()
    v, o = gnet(imgs)
    print(v.shape, o.shape)