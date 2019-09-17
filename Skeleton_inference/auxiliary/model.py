from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torch.nn.functional as F


# UTILITIES
class STN3d(nn.Module):
    def __init__(self, num_points=2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, trans=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x


# OUR METHOD
import resnet


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class SVR_SurSkeNet(nn.Module):
    def __init__(self, num_points=2048, bottleneck_size=1024, nb_primitives=5, pretrained_encoder=False, cuda=True):
        super(SVR_SurSkeNet, self).__init__()
        self.usecuda = cuda
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=1024)
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.nb_primitives)])

    def forward(self, x):
        x = x[:, :3, :, :].contiguous()
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.nb_primitives))
            rand_grid.data.uniform_(0, 1)

            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def decode(self, x):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.nb_primitives))
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            if self.usecuda:
                rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            else:
                rand_grid = Variable(torch.FloatTensor(grid[i]))

            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)

            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()


class SVR_CurSkeNet(nn.Module):
    def __init__(self, num_points=2048, bottleneck_size=1024, nb_primitives=5, pretrained_encoder=False, cuda=True):
        super(SVR_CurSkeNet, self).__init__()
        self.usecuda = cuda
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=1024)
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.nb_primitives)])

    def forward(self, x):
        x = x[:, :3, :, :].contiguous()
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 1, self.num_points // self.nb_primitives))
            rand_grid.data.uniform_(0, 1)

            rand_zero = Variable(torch.zeros(x.size(0), 1, self.num_points // self.nb_primitives).cuda())
            rand_grid = torch.cat((rand_grid, rand_zero), 1).contiguous()

            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            if self.usecuda:
                rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            else:
                rand_grid = Variable(torch.FloatTensor(grid[i]))

            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)

            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()


class AE_SurSkeNet(nn.Module):
    def __init__(self, num_points=2048, bottleneck_size=1024, nb_primitives=1):
        super(AE_SurSkeNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
            PointNetfeat(num_points, global_feat=True, trans=False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.nb_primitives)])

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.nb_primitives))
            rand_grid.data.uniform_(0, 1)

            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)

            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()


class AE_CurSkeNet(nn.Module):
    def __init__(self, num_points=2048, bottleneck_size=1024, nb_primitives=1):
        super(AE_CurSkeNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
            PointNetfeat(num_points, global_feat=True, trans=False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.nb_primitives)])

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 1, self.num_points // self.nb_primitives))
            rand_grid.data.uniform_(0, 1)

            rand_zero = Variable(torch.zeros(x.size(0), 1, self.num_points // self.nb_primitives).cuda())
            rand_grid = torch.cat((rand_grid, rand_zero), 1).contiguous()

            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y.type_as(rand_grid)), 1).contiguous()
        outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))

            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)

            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()


#####################################################################################################################
class SVR_AtlasNet_skeleton(nn.Module):
    def __init__(self, num_line_points=2048, num_square_points=2048, bottleneck_size=1024, nb_primitives_line=5,
                 nb_primitives_square=5, pretrained_encoder=False, cuda=True):
        super(SVR_AtlasNet_skeleton, self).__init__()
        self.usecuda = cuda
        self.num_line_points = num_line_points
        self.num_square_points = num_square_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives_line = nb_primitives_line
        self.nb_primitives_square = nb_primitives_square
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=1024)
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in
                                      range(0, self.nb_primitives_line + self.nb_primitives_square)])

    def forward(self, x):
        x = x[:, :3, :, :].contiguous()
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives_square):
            rand_grid = Variable(
                torch.cuda.FloatTensor(x.size(0), 2, self.num_square_points // self.nb_primitives_square))
            rand_grid.data.uniform_(0, 1)

            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))

        for i in range(0, self.nb_primitives_line):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 1, self.num_line_points // self.nb_primitives_line))
            rand_grid.data.uniform_(0, 1)

            rand_zero = Variable(torch.zeros(x.size(0), 1, self.num_line_points // self.nb_primitives_line).cuda())
            rand_grid = torch.cat((rand_grid, rand_zero), 1).contiguous()

            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[self.nb_primitives_square + i](y))

        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives_square + self.nb_primitives_line):
            if self.usecuda:
                rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            else:
                rand_grid = Variable(torch.FloatTensor(grid[i]))

            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)

            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))

        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()


class AE_AtlasNet_skeleton(nn.Module):
    def __init__(self, num_points=2048, num_line_points=2048, num_square_points=2048, bottleneck_size=1024,
                 nb_primitives_line=5, nb_primitives_square=5):
        super(AE_AtlasNet_skeleton, self).__init__()
        self.num_points = num_points
        self.num_line_points = num_line_points
        self.num_square_points = num_square_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives_line = nb_primitives_line
        self.nb_primitives_square = nb_primitives_square
        self.encoder = nn.Sequential(
            PointNetfeat(num_points, global_feat=True, trans=False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in
                                      range(0, self.nb_primitives_line + self.nb_primitives_square)])

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives_square):
            rand_grid = Variable(
                torch.cuda.FloatTensor(x.size(0), 2, self.num_square_points // self.nb_primitives_square))
            rand_grid.data.uniform_(0, 1)

            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))

        for i in range(0, self.nb_primitives_line):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 1, self.num_line_points // self.nb_primitives_line))
            rand_grid.data.uniform_(0, 1)

            rand_zero = Variable(torch.zeros(x.size(0), 1, self.num_line_points // self.nb_primitives_line))
            rand_grid = torch.cat((rand_grid, rand_zero), 1).contiguous()

            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))

        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives_square + self.nb_primitives_line):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)

            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()


###########################################################################################################
if __name__ == '__main__':

    print('testing SVR_SurSkeNet...')
    sim_data = Variable(torch.rand(1, 3, 224, 224))
    model = SVR_SurSkeNet().cuda()
    out = model(sim_data.cuda())
    print(out.size())
