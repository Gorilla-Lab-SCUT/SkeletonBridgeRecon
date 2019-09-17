from __future__ import print_function
import numpy as np
from layers import Unpool3DLayer, SoftmaxWithLoss3D, WeightSoftmaxWithLoss3D
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Linear, Conv2d, MaxPool2d, \
    LeakyReLU, Conv3d, Tanh, Sigmoid, ReLU
import torch.nn.functional as F
import resnet


class embedding(nn.Module):
    def __init__(self, bottleneck_size=1024):
        super(embedding, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.n_gru_vox = 2
        self.gf_dim = 128
        self.fc = nn.Linear(self.bottleneck_size, self.gf_dim * self.n_gru_vox * self.n_gru_vox * self.n_gru_vox)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.fc(x)
        x = x.view(-1, self.gf_dim, self.n_gru_vox, self.n_gru_vox, self.n_gru_vox)
        return x


class decoder(nn.Module):
    def __init__(self):
        print("\ninitializing \"decoder\"")
        super(decoder, self).__init__()
        self.n_deconvfilter = [128, 128, 128, 64, 32, 2]

        # 3d conv1
        conv1_kernel_size = 3
        self.conv1 = Conv3d(in_channels=self.n_deconvfilter[0],
                            out_channels=self.n_deconvfilter[1],
                            kernel_size=conv1_kernel_size,
                            padding=int((conv1_kernel_size - 1) / 2))

        # 3d conv2
        conv2_kernel_size = 3
        self.conv2 = Conv3d(in_channels=self.n_deconvfilter[1],
                            out_channels=self.n_deconvfilter[2],
                            kernel_size=conv2_kernel_size,
                            padding=int((conv2_kernel_size - 1) / 2))

        # 3d conv3
        conv3_kernel_size = 3
        self.conv3 = Conv3d(in_channels=self.n_deconvfilter[2],
                            out_channels=self.n_deconvfilter[3],
                            kernel_size=conv3_kernel_size,
                            padding=int((conv3_kernel_size - 1) / 2))

        # 3d conv4
        conv4_kernel_size = 3
        self.conv4 = Conv3d(in_channels=self.n_deconvfilter[3],
                            out_channels=self.n_deconvfilter[4],
                            kernel_size=conv4_kernel_size,
                            padding=int((conv4_kernel_size - 1) / 2))

        # 3d conv5
        conv5_kernel_size = 3
        self.conv5 = Conv3d(in_channels=self.n_deconvfilter[4],
                            out_channels=self.n_deconvfilter[5],
                            kernel_size=conv5_kernel_size,
                            padding=int((conv5_kernel_size - 1) / 2))
        # pooling layer
        self.unpool3d = Unpool3DLayer(unpool_size=2)

        # nonlinearities of the network
        self.leaky_relu = LeakyReLU(negative_slope=0.01)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        gru_out = nn.Sequential(self.unpool3d, self.conv1, self.leaky_relu,
                                self.unpool3d, self.conv2, self.leaky_relu,
                                self.unpool3d, self.conv3, self.leaky_relu,
                                self.unpool3d, self.conv4, self.leaky_relu)
        feat = gru_out(x)
        x = self.conv5(feat)
        occupany = x
        return feat, occupany


class SVR_R2N2(nn.Module):
    def __init__(self, voxel_size=32, bottleneck_size=1024, pretrained_encoder=False):  # 1024
        super(SVR_R2N2, self).__init__()
        self.voxel_size = voxel_size
        self.bottleneck_size = bottleneck_size
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=self.bottleneck_size)
        self.embedding = embedding(bottleneck_size=self.bottleneck_size)
        self.decoder = decoder()

    def forward(self, x):
        x = x[:, :3, :, :].contiguous()
        x = self.encoder(x)
        x = self.embedding(x)
        feat, occupany = self.decoder(x)
        return feat, occupany


class Encoder32(nn.Module):
    def __init__(self):
        super(Encoder32, self).__init__()
        self.conv0_0 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv0_1 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_0 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_0 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)

        self.conv0_0_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(16)
        self.conv1_0_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(32)
        self.conv2_0_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(64)
        self.conv3_bn = nn.BatchNorm3d(128)
        self.maxpool = nn.MaxPool3d(2, return_indices=True)

    def encoder(self, x):
        x = F.relu(self.conv0_0_bn(self.conv0_0(x)))
        x = F.relu(self.conv0_1_bn(self.conv0_1(x)))
        feat0 = x
        size0 = x.size()
        x, indices0 = self.maxpool(x)

        x = F.relu(self.conv1_0_bn(self.conv1_0(x)))
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        feat1 = x
        size1 = x.size()
        x, indices1 = self.maxpool(x)

        x = F.relu(self.conv2_0_bn(self.conv2_0(x)))
        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        feat2 = x
        feat2 = x
        size2 = x.size()
        x, indices2 = self.maxpool(x)

        x = F.relu(self.conv3_bn(self.conv3(x)))
        return x, feat2, size2, indices2, feat1, size1, indices1, feat0, size0, indices0

    def forward(self, x):
        x, feat2, size2, indices2, feat1, size1, indices1, feat0, size0, indices0 = self.encoder(x)
        return x, feat2, size2, indices2, feat1, size1, indices1, feat0, size0, indices0


class Decoder32(nn.Module):
    def __init__(self):
        super(Decoder32, self).__init__()
        self.deconv3 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2_1 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2_0 = nn.Conv3d(128, 32, kernel_size=3, stride=1, padding=1)
        self.deconv1_1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv1_0 = nn.Conv3d(64, 16, kernel_size=3, stride=1, padding=1)
        self.deconv0_1 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv_cat = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deconv0_0 = nn.Conv3d(32, 2, kernel_size=3, stride=1, padding=1)

        self.deconv3_bn = nn.BatchNorm3d(64)
        self.deconv2_1_bn = nn.BatchNorm3d(128)
        self.deconv2_0_bn = nn.BatchNorm3d(32)
        self.deconv1_1_bn = nn.BatchNorm3d(64)
        self.deconv1_0_bn = nn.BatchNorm3d(16)
        self.deconv0_1_bn = nn.BatchNorm3d(32)
        self.deconv_cat_bn = nn.BatchNorm3d(32)

        self.maxunpool = nn.MaxUnpool3d(2)
        self.log_softmax = nn.LogSoftmax()

    def decoder(self, x, feat2, size2, indices2, feat1, size1, indices1, feat0, size0, indices0, feat):
        x = F.relu(self.deconv3_bn(self.deconv3(x)))

        x = self.maxunpool(x, indices2, output_size=size2)
        x = torch.cat((feat2, x), 1)
        x = F.relu(self.deconv2_1_bn(self.deconv2_1(x)))
        x = F.relu(self.deconv2_0_bn(self.deconv2_0(x)))

        x = self.maxunpool(x, indices1, output_size=size1)
        x = torch.cat((feat1, x), 1)
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(x)))
        x = F.relu(self.deconv1_0_bn(self.deconv1_0(x)))

        x = self.maxunpool(x, indices0, output_size=size0)
        x = torch.cat((feat0, x), 1)
        x = F.relu(self.deconv0_1_bn(self.deconv0_1(x)))

        x = torch.cat((feat, x), 1)
        feat = F.relu(self.deconv_cat_bn(self.deconv_cat(x)))
        x = self.deconv0_0(feat)
        occupany = x
        return feat, occupany

    def forward(self, x, feat2, size2, indices2, feat1, size1, indices1, feat0, size0, indices0, feat):
        feat, occupany = self.decoder(x, feat2, size2, indices2, feat1, size1, indices1, feat0, size0, indices0, feat)
        return feat, occupany


class Encoder64(nn.Module):
    def __init__(self):
        super(Encoder64, self).__init__()
        self.conv0_0 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv0_1 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv1_0 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
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

    def encoder(self, x):
        x = F.relu(self.conv0_0_bn(self.conv0_0(x)))
        x = F.relu(self.conv0_1_bn(self.conv0_1(x)))
        feat0 = x

        x = F.relu(self.conv1_0_bn(self.conv1_0(x)))
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        feat1 = x

        x = F.relu(self.conv2_0_bn(self.conv2_0(x)))
        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        feat2 = x

        x = F.relu(self.conv3_bn(self.conv3(x)))
        return x, feat2, feat1, feat0

    def forward(self, x):
        x, feat2, feat1, feat0 = self.encoder(x)
        return x, feat2, feat1, feat0


class Decoder64(nn.Module):
    def __init__(self):
        super(Decoder64, self).__init__()
        self.deconv3 = nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2_1 = nn.ConvTranspose3d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2_0 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv1_1 = nn.ConvTranspose3d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1_0 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv0_1 = nn.ConvTranspose3d(96, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv0_0 = nn.Conv3d(16, 2, kernel_size=3, stride=1, padding=1)

        self.deconv3_bn = nn.BatchNorm3d(128)
        self.deconv2_1_bn = nn.BatchNorm3d(64)
        self.deconv2_0_bn = nn.BatchNorm3d(64)
        self.deconv1_1_bn = nn.BatchNorm3d(32)
        self.deconv1_0_bn = nn.BatchNorm3d(32)
        self.deconv0_1_bn = nn.BatchNorm3d(16)

        self.log_softmax = nn.LogSoftmax()

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


class Global_Guidance(nn.Module):
    def __init__(self):
        super(Global_Guidance, self).__init__()
        self.svr = SVR_R2N2()
        self.encoder32 = Encoder32()
        self.decoder32 = Decoder32()
        self.encoder64 = Encoder64()
        self.decoder64 = Decoder64()

    def forward(self, img, input1, input2):
        # img reconstruction 32
        feat, occupany1 = self.svr(img)
        # input 32 refinement
        x, feat2, size2, indices2, feat1, size1, indices1, feat0, size0, indices0 = self.encoder32(input1)
        feat, occupany2 = self.decoder32(x, feat2, size2, indices2, feat1, size1, indices1, feat0, size0, indices0, feat)
        # input 64 refienment
        x, feat2, feat1, feat0 = self.encoder64(input2)
        occupany3 = self.decoder64(x, feat2, feat1, feat0, feat)
        return occupany1, occupany2, occupany3

if __name__ == '__main__':
    print("Testing CNN3d_Unet_Global")
    img = Variable(torch.rand(1, 3, 224, 224))
    input1 = Variable(torch.randn(1, 1, 32, 32, 32))
    input2 = Variable(torch.randn(1, 1, 64, 64, 64))
    model1 = Global_Guidance()
    model1.cuda()
    occupany1, occupany2, occupany3 = model1(img.cuda(), input1.cuda(), input2.cuda())
    print('occupany1, occupany2, occupany3:', occupany1.size(), occupany2.size(), occupany3.size())
