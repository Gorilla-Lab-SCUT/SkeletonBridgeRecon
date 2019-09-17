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

class Local_ShapeNet(data.Dataset):
    '''
    input: patch 64 resolution 128
    target: patch 128 resolution 128
    '''
    def __init__(self, input_root='./data',
                 gt_root='./data',
                 filelist_root='./data/split_train_test',
                 cat='03001627', filelist=None, train=True):

        self.input_root = input_root
        self.gt_root = gt_root
        self.filelist_root = filelist_root
        self.cat = cat

        self.gt_dir = os.path.join(self.gt_root, self.cat, 'skeleton_binvox128_gt')
        if train:
            self.refine_dir = os.path.join(self.gt_root, self.cat, 'after_refinement_binvox64')
            self.input_dir = os.path.join(self.input_root, self.cat, 'skeleton_prediction_binvox128_train')
        else:
            self.refine_dir = os.path.join(self.gt_root, self.cat, 'after_refinement_binvox64')
            self.input_dir = os.path.join(self.input_root, self.cat, 'skeleton_prediction_binvox128_test')

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

        self.idx = 0
        self.data_paths = []
        for f in fnames:
            mod, seq = f.split()
            refine_path = os.path.join(self.refine_dir, mod + '_' + seq + '.binvox')
            input_path = os.path.join(self.input_dir, mod + '_' + seq + '.binvox')
            gt_path = os.path.join(self.gt_dir, mod + '_' + seq + '.binvox')
            self.data_paths.append((refine_path, input_path, gt_path, f, mod, seq))

        # shuffle data_paths to get random patches
        self.random_data_paths = []
        for f in fnames:
            mod, seq = f.split()
            refine_path = os.path.join(self.refine_dir, mod + '_' + seq + '.binvox')
            input_path = os.path.join(self.input_dir, mod + '_' + seq + '.binvox')
            gt_path = os.path.join(self.gt_dir, mod + '_' + seq + '.binvox')
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        self.random_data_paths.append((refine_path, input_path, gt_path, i, j, k, f, mod, seq))
        np.random.shuffle(self.random_data_paths)

    def get_batch(self):
        if self.idx >= len(self.data_paths):
            self.idx = 0
        refine_path, input_path, gt_path, f, mod, seq = self.data_paths[self.idx]
        fp = open(refine_path, 'rb')
        refine = binvox_rw.read_as_3d_array(fp, fix_coords=False).data
        refine = refine.astype('float32')

        fp = open(input_path, 'rb')
        input = binvox_rw.read_as_3d_array(fp, fix_coords=False).data
        input = input.astype('float32')

        fp = open(gt_path, 'rb')
        voxel_data = binvox_rw.read_as_3d_array(fp, fix_coords=False).data
        gt = np.zeros((128, 128, 128), dtype='int64')
        gt[:] = voxel_data

        refine_batch = []
        input_batch = []
        gt_batch = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    refine_batch.append(refine[None, None, i * 32:(i + 1) * 32, j * 32:(j + 1) * 32, k * 32:(k + 1) * 32])
                    input_batch.append(input[None, None, i * 64:(i + 1) * 64, j* 64:(j + 1) * 64, k * 64:(k + 1) * 64])
                    gt_batch.append(gt[None, i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, k * 64:(k + 1) * 64])
        refine_batch = np.concatenate(refine_batch, axis=0)
        input_batch = np.concatenate(input_batch, axis=0)
        gt_batch = np.concatenate(gt_batch, axis=0)
        refine_batch = torch.from_numpy(refine_batch).type(torch.FloatTensor)
        input_batch = torch.from_numpy(input_batch).type(torch.FloatTensor)
        gt_batch = torch.from_numpy(gt_batch).type(torch.LongTensor)
        self.idx += 1
        return refine_batch, input_batch, gt_batch, f, mod, seq

    # get random patches
    def __getitem__(self, index):
        refine_path, input_path, gt_path, i, j, k, f, mod, seq = self.random_data_paths[index]
        fp = open(refine_path, 'rb')
        refine = binvox_rw.read_as_3d_array(fp, fix_coords=False).data
        refine = refine.astype('float32')

        fp = open(input_path, 'rb')
        input = binvox_rw.read_as_3d_array(fp, fix_coords=False).data
        input = input.astype('float32')

        fp = open(gt_path, 'rb')
        voxel_data = binvox_rw.read_as_3d_array(fp, fix_coords=False).data
        gt = np.zeros((128, 128, 128), dtype='int64')
        gt[:] = voxel_data

        refine = refine[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32, k * 32:(k + 1) * 32]
        input = input[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, k * 64:(k + 1) * 64]
        gt = gt[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, k * 64:(k + 1) * 64]
        refine = torch.from_numpy(refine[None, :, :, :]).type(torch.FloatTensor)
        input = torch.from_numpy(input[None, :, :, :]).type(torch.FloatTensor)
        gt = torch.from_numpy(gt).type(torch.LongTensor)
        return refine, input, gt, i, j, k, f, mod, seq

    def __len__(self):
        return len(self.random_data_paths)

if __name__ == '__main__':
    dataset = Local_ShapeNet(train=True, cat='04379243')
    for i in range(0, len(dataset)):
        refine, input, gt, f, mod, seq = dataset.get_batch()
        print(i, refine.size(), input.size(), gt.size(), f)

    dataset = Local_ShapeNet(train=False, cat='04379243')
    for i in range(0, len(dataset)):
        refine, input, gt, f, mod, seq = dataset.get_batch()
        print(i, refine.size(), input.size(), gt.size(), f)

    dataset = Local_ShapeNet(train=True, cat='04379243')
    for idx in range(0, len(dataset)):
        refine, input, gt, i, j, k, f, mod, seq = dataset[i]
        print(idx, refine.size(), input.size(), gt.size(), f, i, j, k)

    dataset = Local_ShapeNet(train=False, cat='04379243')
    for idx in range(0, len(dataset)):
        refine, input, gt, f, i, j, k, mod, seq = dataset[i]
        print(idx, refine.size(), input.size(), gt.size(), f, i, j, k)
