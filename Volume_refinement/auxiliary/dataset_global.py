import numpy as np
import pickle
import scipy.sparse as sp
import os
import sys
import math
import time
import random
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from PIL import Image
import binvox_rw


class Global_ShapeNet(data.Dataset):
    def __init__(self, rgb_root='/data1/tang.jiapeng/ShapeNetRendering/', #change the shapenet rendeing path
                 input_root='./data',
                 gt_root='./data',
                 filelist_root='./data/split_train_test',
                 cat='03001627', filelist=None, train=True):

        self.rgb_root = rgb_root
        self.input_root = input_root
        self.gt_root = gt_root
        self.filelist_root = filelist_root
        self.cat = cat
        self.train = train

        self.rgb_dir = os.path.join(self.rgb_root, self.cat)
        self.gt32_dir = os.path.join(self.gt_root, self.cat, 'skeleton_binvox32_gt')
        self.gt64_dir = os.path.join(self.gt_root, self.cat, 'skeleton_binvox64_gt')
        if train:
            self.input32_dir = os.path.join(self.input_root, self.cat, 'skeleton_prediction_binvox32_train')
            self.input64_dir = os.path.join(self.input_root, self.cat, 'skeleton_prediction_binvox64_train')
        else:
            self.input32_dir = os.path.join(self.input_root, self.cat, 'skeleton_prediction_binvox32_test')
            self.input64_dir = os.path.join(self.input_root, self.cat, 'skeleton_prediction_binvox64_test')

        if train:
            self.filelist = 'trainlist_index.txt'
        else:
            if filelist is None:
                self.filelist = 'testlist_index.txt'
            else:
                self.filelist = filelist
        self.filelist_path = os.path.join(self.filelist_root, self.cat, self.filelist)
        fnames = open(self.filelist_path, 'r').readlines()
        fnames = [f.strip() for f in fnames]

        self.data_paths = []
        for f in fnames:
            mod, seq = f.split()
            rgb_path = os.path.join(self.rgb_dir, mod, 'rendering', seq + '.png')
            input32_path = os.path.join(self.input32_dir, mod + '_' + seq + '.binvox')
            gt32_path = os.path.join(self.gt32_dir, mod + '_' + seq + '.binvox')
            input64_path = os.path.join(self.input64_dir, mod + '_' + seq + '.binvox')
            gt64_path = os.path.join(self.gt64_dir, mod + '_' + seq + '.binvox')
            self.data_paths.append((rgb_path, input32_path, gt32_path, input64_path, gt64_path, f, mod, seq))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transforms = transforms.Compose([
            transforms.Scale(size=224, interpolation=2),
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

    def __getitem__(self, index):
        rgb_path, input32_path, gt32_path, input64_path, gt64_path, f, mod, seq = self.data_paths[index]
        if self.train:
            im = Image.open(rgb_path)
            im = self.dataAugmentation(im)  # random crop
        else:
            im = Image.open(rgb_path)
            im = self.validating(im)  # center crop
        data = self.transforms(im)  # scale
        data = data[:3, :, :]

        fp = open(input32_path, 'rb')
        input32 = binvox_rw.read_as_3d_array(fp, fix_coords=False).data
        input32 = input32.astype('float32')
        input32 = torch.from_numpy(input32[None, :, :, :]).type(torch.FloatTensor)

        fp = open(gt32_path, 'rb')
        voxel_data = binvox_rw.read_as_3d_array(fp, fix_coords=False).data
        gt32 = np.zeros((32, 32, 32), dtype='int64')
        gt32[:] = voxel_data
        gt32 = torch.from_numpy(gt32).type(torch.LongTensor)

        fp = open(input64_path, 'rb')
        input64 = binvox_rw.read_as_3d_array(fp, fix_coords=False).data
        input64 = input64.astype('float32')
        input64 = torch.from_numpy(input64[None, :, :, :]).type(torch.FloatTensor)

        fp = open(gt64_path, 'rb')
        voxel_data = binvox_rw.read_as_3d_array(fp, fix_coords=False).data
        gt64 = np.zeros((64, 64, 64), dtype='int64')
        gt64[:] = voxel_data
        gt64 = torch.from_numpy(gt64).type(torch.LongTensor)
        return data, input32, gt32, input64, gt64, f, mod, seq

    def __len__(self):
        return len(self.data_paths)

if __name__ == '__main__':
    dataset = Global_ShapeNet(train=True)
    for i in range(0, len(dataset)):
        data, input32, gt32, input64, gt64, f, mod, seq = dataset[i]
        print(i, data.size(), input32.size(), gt32.size(), input64.size(), gt64.size(), f)

    dataset = Global_ShapeNet(train=False)
    for i in range(0, len(dataset)):
        data, input32, gt32, input64, gt64, f, mod, seq = dataset[i]
        print(i, data.size(), input32.size(), gt32.size(), input64.size(), gt64.size(), f)
