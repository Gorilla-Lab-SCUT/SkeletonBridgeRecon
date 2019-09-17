from __future__ import print_function
import torch.utils.data as data
import os.path
import errno
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import math
import os
import sys
import scipy.io as sio
from PIL import Image
from utils import *


class ShapeNet(data.Dataset):
    def __init__(self, rootimg="/data1/tang.jiapeng/ShapeNetRendering",
                 rootpc="/data1/tang.jiapeng/ShapeNetPointCloud",
                 class_choice="chair", train=True, npoints_line=2500, npoints_square=5000,
                 normal=False, SVR=False):
        self.normal = normal
        self.train = train
        self.rootimg = rootimg
        self.rootpc = rootpc
        self.npoints_line = npoints_line
        self.npoints_square = npoints_square
        self.datapath = []
        self.catfile = os.path.join('./data/synsetoffset2category.txt')
        self.cat = {}
        self.meta = {}
        self.SVR = SVR
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        print(self.cat)
        empty = []
        for item in self.cat:
            dir_img = os.path.join(self.rootimg, self.cat[item])
            fns_img = open(os.path.join('./data/split_train_test', self.cat[item], 'all.txt'),
                           'r').readlines()
            fns_img = [i.strip() for i in fns_img]

            try:
                dir_point = os.path.join(self.rootpc, self.cat[item], 'classify_mat')
                fns_pc = sorted(os.listdir(dir_point))
            except:
                fns_pc = []
            fns = [val for val in fns_img if val + '_line.mat' in fns_pc and val + '_square.mat' in fns_pc]
            print('category ', self.cat[item], 'files ' + str(len(fns)), len(fns) / float(len(fns_img)), "%"),

            if train:
                fns = open(os.path.join('./data/split_train_test', self.cat[item], 'trainlist_index.txt'),
                           'r').readlines()
                fns = [fn.strip() for fn in fns]
            else:
                fns = open(os.path.join('./data/split_train_test', self.cat[item], 'testlist_index.txt'),
                           'r').readlines()
                fns = [fn.strip() for fn in fns]
            if len(fns) != 0:
                self.meta[item] = []
            for fn in fns:
                modname, index = fn.split()
                img_path = os.path.join(dir_img, modname, "rendering", index + '.png')
                line_path = os.path.join(dir_point, modname + '_line.mat')
                square_path = os.path.join(dir_point, modname + '_square.mat')
                para_path = os.path.join(dir_img, modname, "rendering", "rendering_metadata.txt")
                params = open(para_path).readlines()[int(index)]
                azimuth, elevation, _, distance, _ = map(float, params.strip().split())
                self.meta[item].append(
                    (img_path, line_path, item, square_path, modname, (azimuth, elevation, distance), index))
        for item in empty:
            del self.cat[item]
        self.idx2cat = {}
        self.size = {}
        i = 0
        for item in self.cat:
            self.idx2cat[i] = item
            self.size[i] = len(self.meta[item])
            i = i + 1
            for fn in self.meta[item]:
                self.datapath.append(fn)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transforms = transforms.Compose([
            transforms.Resize(size=224, interpolation=2),
            transforms.ToTensor(),
            # normalize,
        ])

        # RandomResizedCrop or RandomCrop
        self.dataAugmentation = transforms.Compose([
            transforms.RandomCrop(127),
            transforms.RandomHorizontalFlip(),
        ])
        self.validating = transforms.Compose([
            transforms.CenterCrop(127),
        ])

        self.perCatValueMeter = {}
        for item in self.cat:
            self.perCatValueMeter[item] = AverageValueMeter()
        self.transformsb = transforms.Compose([
            transforms.Resize(size=224, interpolation=2),
        ])

    def __getitem__(self, index):
        fn = self.datapath[index]
        point_set_line = sio.loadmat(fn[1])['v']
        point_set_square = sio.loadmat(fn[3])['v']
        azimuth, elevation, distance = fn[5]

        azi = math.radians(azimuth)
        ele = math.radians(elevation)
        dis = distance
        eye = (dis * math.cos(ele) * math.cos(azi),
               dis * math.sin(ele),
               dis * math.cos(ele) * math.sin(azi))
        eye = np.asarray(eye)
        at = np.array([0, 0, 0], dtype='float32')
        up = np.array([0, 1, 0], dtype='float32')
        z_axis = normalize(eye - at, eps=1e-5)  # forward
        x_axis = normalize(np.cross(up, z_axis), eps=1e-5)  # left
        y_axis = normalize(np.cross(z_axis, x_axis), eps=1e-5)  # up
        # rotation matrix: [3, 3]
        R = np.concatenate((x_axis[None, :], y_axis[None, :], z_axis[None, :]), axis=0)

        point_set_line = point_set_line.dot(R.T)
        point_set_square = point_set_square.dot(R.T)
        if len(point_set_line) <= self.npoints_line:
            times = self.npoints_line / len(point_set_line)
            point_set_line = np.repeat(point_set_line, times, 0)
            left = self.npoints_line - len(point_set_line)
            point_left = point_set_line[np.random.choice(point_set_line.shape[0], left)]
            point_set_line = np.concatenate((point_set_line, point_left), axis=0)
        else:
            point_set_line = point_set_line[np.random.choice(point_set_line.shape[0], self.npoints_line)]

        if len(point_set_square) <= self.npoints_square:
            times = self.npoints_square / len(point_set_square)
            point_set_square = np.repeat(point_set_square, times, 0)
            left = self.npoints_square - len(point_set_square)
            point_left = point_set_square[np.random.choice(point_set_square.shape[0], left)]
            point_set_square = np.concatenate((point_set_square, point_left), axis=0)
        else:
            point_set_square = point_set_square[np.random.choice(point_set_square.shape[0], self.npoints_square)]

        point_set_line = torch.from_numpy(point_set_line).type(torch.FloatTensor)
        point_set_square = torch.from_numpy(point_set_square).type(torch.FloatTensor)

        # load image
        if self.SVR:
            if self.train:
                im = Image.open(fn[0])
                im = self.dataAugmentation(im)  # random crop
            else:
                im = Image.open(fn[0])
                im = self.validating(im)  # center crop
            data = self.transforms(im)  # Resize
            data = data[:3, :, :]
        else:
            data = 0
        return data, point_set_line.contiguous(), fn[2], point_set_square.contiguous(), fn[4], fn[6]
        # 3*224*224 2500*3 5000*3

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':

    print('Testing Shapenet dataset')
    d1 = ShapeNet(class_choice="chair", train=True, SVR=True)
    a = len(d1)
    d2 = ShapeNet(class_choice="chair", train=False, SVR=True)
    a = a + len(d2)
    dataloader = torch.utils.data.DataLoader(d2, batch_size=4, shuffle=False,
                                             num_workers=int(opt.workers))  # shuffle??
    print('length of set ', a)
    for i, data in enumerate(dataloader, 0):
        print(data[0], data[1], data[2], data[3], data[4], data[5])
        print(data[1].size(), data[3].size())
        if i>1:
            break
