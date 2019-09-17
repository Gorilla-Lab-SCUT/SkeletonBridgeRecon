from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------------------------------#
class Encoder_Patch64(nn.Module):
    def __init__(self):
        super(Encoder_Patch64, self).__init__()
        self.conv0_0 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv0_1 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv1_0 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_0 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)

        self.conv0_0_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_0_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_0_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_bn = nn.BatchNorm3d(256)

    def encoder(self, x, feat):
        x = F.relu(self.conv0_0_bn(self.conv0_0(x)))
        x = F.relu(self.conv0_1_bn(self.conv0_1(x)))
        feat0 = x

        x = torch.cat((x, feat), 1)
        x = F.relu(self.conv1_0_bn(self.conv1_0(x)))
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        feat1 = x

        x = F.relu(self.conv2_0_bn(self.conv2_0(x)))
        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        feat2 = x

        x = F.relu(self.conv3_bn(self.conv3(x)))
        return x, feat2, feat1, feat0

    def forward(self, x, feat):
        x, feat2, feat1, feat0 = self.encoder(x, feat)
        return x, feat2, feat1, feat0


class Decoder_Patch64(nn.Module):
    def __init__(self):
        super(Decoder_Patch64, self).__init__()
        self.deconv3 = nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2_1 = nn.ConvTranspose3d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2_0 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv1_1 = nn.ConvTranspose3d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1_0 = nn.ConvTranspose3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv0_1 = nn.ConvTranspose3d(96, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv0_0 = nn.ConvTranspose3d(16, 2, kernel_size=3, stride=1, padding=1)

        self.deconv3_bn = nn.BatchNorm3d(128)
        self.deconv2_1_bn = nn.BatchNorm3d(64)
        self.deconv2_0_bn = nn.BatchNorm3d(64)
        self.deconv1_1_bn = nn.BatchNorm3d(32)
        self.deconv1_0_bn = nn.BatchNorm3d(32)
        self.deconv0_1_bn = nn.BatchNorm3d(16)

    def decoder(self, x, feat2, feat1, feat0, feat):
        x = F.relu(self.deconv3_bn(self.deconv3(x)))

        x = torch.cat((feat2, x), 1)
        x = F.relu(self.deconv2_1_bn(self.deconv2_1(x)))
        x = F.relu(self.deconv2_0_bn(self.deconv2_0(x)))

        x = torch.cat((feat1, x), 1)
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(x)))
        x = F.relu(self.deconv1_0_bn(self.deconv1_0(x)))

        x = torch.cat((feat, feat0, x), 1)
        x = F.relu(self.deconv0_1_bn(self.deconv0_1(x)))
        x = self.deconv0_0(x)
        occupany = x
        return occupany

    def forward(self, x, feat2, feat1, feat0, feat):
        occupany = self.decoder(x, feat2, feat1, feat0, feat)
        return occupany


class Local_Synthesis(nn.Module):
    def __init__(self):
        super(Local_Synthesis, self).__init__()
        self.conv0_0 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv0_1 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_0 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv0_0_bn = nn.BatchNorm3d(32)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_0_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(32)
        self.encoder = Encoder_Patch64()
        self.decoder = Decoder_Patch64()

    def forward(self, x1, x2):
        feat_conv0 = F.relu(self.conv0_0_bn(self.conv0_0(x1)))
        feat_conv0 = F.relu(self.conv0_1_bn(self.conv0_1(feat_conv0)))
        feat_conv1 = F.relu(self.conv1_0_bn(self.conv1_0(x1)))
        feat_conv1 = F.relu(self.conv1_1_bn(self.conv1_1(feat_conv1)))
        x, feat2, feat1, feat0 = self.encoder(x2, feat_conv0)
        occupany = self.decoder(x, feat2, feat1, feat0, feat_conv1)
        return occupany

if __name__ == '__main__':
    print("Testing CNN3d_Unet_Local")
    input1 = Variable(torch.randn(1, 1, 64, 64, 64))
    input2 = Variable(torch.randn(8, 1, 64, 64, 64))
    model2 = Local_Synthesis()
    model2.cuda()
    occupany = model2(input1.cuda(), input2.cuda())
    print('occupany:', occupany.size())