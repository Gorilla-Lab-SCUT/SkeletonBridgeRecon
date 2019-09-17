from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import os
import json
import time, datetime
import visdom
sys.path.append('./auxiliary/')
from layers import *
from dataset_local import *
from model_local import *
from utils import *
import binvox_rw
sys.path.append('./external/')
import libmcubes
import libsimplify
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--th', type=float, default=0.4, help='the thresold to compute IoU')
parser.add_argument('--category', type=str, default='chair')
parser.add_argument('--start', type=int, default=0, help='start index')
parser.add_argument('--nfaces', type=int, default = 10000,  help='number of faces of base meshes')
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Creat train/test dataloader
dataset = Local_ShapeNet(cat=class_name[opt.category], train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
dataset_test = Local_ShapeNet(cat=class_name[opt.category], train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=int(opt.workers))
print('training samples', len(dataset)/8)
len_dataset = len(dataset)
print('testing samples', len(dataset_test)/8)
len_dataset_test = len(dataset_test)

# Create network
network = Local_Synthesis()
network.apply(weights_init)
network.cuda()
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
else:
    print(" Give your trained model")

outdir = './data/%s/baseMesh_simplify' % class_name[opt.category]
gt_dir = './data/%s/skeleton_binvox32_gt' % class_name[opt.category]
if not os.path.exists(outdir):
    os.mkdir(outdir)

start = opt.start
dataset.idx = start
network.eval()
with torch.no_grad():
    for idx in range(start, len_dataset/8):
        t0 = time.time()
        refine, input, gt, f, mod, seq = dataset.get_batch()
        objfile = os.path.join(outdir, mod + '_' + seq + '.obj')
        if os.path.exists(objfile):
            continue
        refine = refine.cuda()
        input = input.cuda()
        gt = gt.cuda()

        output = network(refine, input)
        output = F.softmax(output, dim=1)
        output = torch.ge(output[:, 1, :, :, :], opt.th).type(torch.cuda.FloatTensor)
        gt = gt.type(torch.cuda.FloatTensor)

        prediction = torch.cuda.FloatTensor(1, 128, 128, 128)
        groudtruth = torch.cuda.FloatTensor(1, 128, 128, 128)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    prediction[:, i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, k * 64:(k + 1) * 64] = output.data[i * 4 + j * 2 + k, :, :, :]
                    groudtruth[:, i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, k * 64:(k + 1) * 64] = gt.data[i * 4 + j * 2 + k, :, :, :]

        objfile = os.path.join(outdir, mod + '_' + seq + '.obj')
        voxels = torch.squeeze(prediction).cpu().numpy()
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))  ## padding for generating a closed shape
        gt_file = os.path.join(gt_dir, mod + '_' + seq + '.binvox')
        with open(gt_file, 'rb') as fp:
            dims, translate, scale = binvox_rw.read_header(fp)
        vertices, faces, = libmcubes.marching_cubes(voxels, 0.5)
        vertices = (vertices - 63.5)/128.0 * scale + np.array([0, 0, -0.8], dtype='f4') 
        #vertices = (vertices - 63.5)/128.0 * scale + translate # don't use this
        mesh = trimesh.Trimesh(vertices, faces, vertex_normals=None, process=False)
        mesh = libsimplify.simplify_mesh(mesh, opt.nfaces)
        mesh.export(objfile)
        print('[%d/%d] time: %f' % (idx, len(dataset) / 8, time.time() - t0), objfile)

network.eval()
if True:
    for idx in range(len_dataset_test/8):
        t0 = time.time()
        refine, input, gt, f, mod, seq = dataset_test.get_batch()
        objfile = os.path.join(outdir, mod + '_' + seq + '.obj')
        if os.path.exists(objfile):
            continue
        refine = refine.cuda()
        input = input.cuda()
        gt = gt.cuda()

        output = network(refine, input)
        output = F.softmax(output, dim=1)
        output = torch.ge(output[:, 1, :, :, :], opt.th).type(torch.cuda.FloatTensor)
        gt = gt.type(torch.cuda.FloatTensor)

        prediction = torch.cuda.FloatTensor(1, 128, 128, 128)
        groudtruth = torch.cuda.FloatTensor(1, 128, 128, 128)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    prediction[:, i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, k * 64:(k + 1) * 64] = output.data[i * 4 + j * 2 + k, :, :, :]
                    groudtruth[:, i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, k * 64:(k + 1) * 64] = gt.data[i * 4 + j * 2 + k, :, :, :]

        objfile = os.path.join(outdir, mod + '_' + seq + '.obj')
        voxels = torch.squeeze(prediction).cpu().numpy()
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))  ## padding for generating a closed shape
        gt_file = os.path.join(gt_dir, mod + '_' + seq + '.binvox')
        with open(gt_file, 'rb') as fp:
            dims, translate, scale = binvox_rw.read_header(fp)
        vertices, faces, = libmcubes.marching_cubes(voxels, 0.5)
        vertices = (vertices - 63.5)/128.0 * scale + np.array([0, 0, -0.8], dtype='f4') 
        ##vertices = (vertices - 63.5)/128.0 * scale + translate #don't use this
        mesh = trimesh.Trimesh(vertices, faces, vertex_normals=None, process=False)
        mesh = libsimplify.simplify_mesh(mesh, opt.nfaces)
        mesh.export(objfile)
        print('[%d/%d] time: %f' % (idx, len(dataset_test) / 8, time.time() - t0), objfile)