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
import sys
sys.path.append('./auxiliary/')
from dataset import *
import binvox_rw
from model import *
from utils import *
from plyio import *
import torch.nn.functional as F
import os
import json
import time, datetime
import subprocess
import pandas as pd
import multiprocessing
import pickle
import scipy.misc


def process_binvox(outfile, res, prediction, fn, idx):
    # output voxel .binvox file
    # computer center and scale
    t = time.time()
    MIN = np.min(prediction, 0)
    MAX = np.max(prediction, 0)
    translate = (MIN + MAX) * 0.5
    translate = [float(x) for x in translate]
    scale = np.max(MAX - MIN)
    dims = [res, res, res]
    order = 'xzy'
    voxel_data = ((prediction - translate) / scale * res + (res - 1.0) / 2.0).T
    with open(outfile, 'wb') as fout:
        binvox_rw.write(binvox_rw.Voxels(np.ascontiguousarray(voxel_data), dims, translate, scale, order), fout)
    print(fn, idx, voxel_data.shape, ' took %f[s]' % (time.time() - t))

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model_line', type=str, default='./trained_models/svr_line20_pretrained.pth',
                    help='your path to the trained model')
parser.add_argument('--model_square', type=str, default='./trained_models/svr_square20_pretrained.pth')
parser.add_argument('--res', type=int, default=128, help='current resolution')
parser.add_argument('--num_points', type=int, default=2500, help='number of points fed to poitnet')
parser.add_argument('--gen_line_points', type=int, default=600, help='number of line points to generate')
parser.add_argument('--gen_square_points', type=int, default=2000, help='number of square points to generate')
parser.add_argument('--nb_primitives_line', type=int, default=20, help='number of primitives of squares')
parser.add_argument('--nb_primitives_square', type=int, default=20, help='number of primitives of squares')
parser.add_argument('--trainset', action='store_true',default=False, help='trainset or testset')
parser.add_argument('--category', type=str, default='chair')
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

catfile = os.path.join('./data/synsetoffset2category.txt')
catidx = {}
with open(catfile, 'r') as f:
    for line in f:
        ls = line.strip().split()
        catidx[ls[0]] = ls[1]
print(catidx)

opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNet(SVR=True, normal=False, class_choice=opt.category, train=opt.trainset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, #shuffle???
                                         num_workers=int(opt.workers))
print('length of set ', len(dataset.datapath))
len_dataset = len(dataset)

vx_res = opt.res

network_line = SVR_CurSkeNet(num_points=opt.gen_line_points, nb_primitives=opt.nb_primitives_line)
network_square = SVR_SurSkeNet(num_points=opt.gen_square_points, nb_primitives=opt.nb_primitives_square)

network_line.cuda()
network_square.cuda()

network_line.apply(weights_init)
network_square.apply(weights_init)

if opt.model_line != '':
    network_line.load_state_dict(torch.load(opt.model_line))
    print("previous model_line weight loaded")
if opt.model_square != '':
    network_square.load_state_dict(torch.load(opt.model_square))
    print("previous model_square weight loaded")

print(network_line)
print(network_square)

train_loss = AverageValueMeter()
val_loss = AverageValueMeter()

network_line.eval()
network_square.eval()

grain = int(np.sqrt(opt.gen_square_points / opt.nb_primitives_square)) - 1
grain = grain * 1.0
print(grain)
grain2 = int(opt.gen_line_points / opt.nb_primitives_line) - 1
grain2 = grain2 * 1.0
print(grain2)

# reset meters
val_loss.reset()
for item in dataset.cat:
    dataset.perCatValueMeter[item].reset()

# generate regular grid
faces = []
vertices = []
vertices2 = []

for i in range(0, int(grain + 1)):
    for j in range(0, int(grain + 1)):
        vertices.append([i / grain, j / grain])

for i in range(0, int(grain2 + 1)):
    vertices2.append([i / grain2, 0])

grid_square = [vertices for i in range(0, opt.nb_primitives_square)]
grid_line = [vertices2 for i in range(0, opt.nb_primitives_line)]

results = dataset.cat.copy()
for i in results:
    results[i] = 0

# Iterate on the data
for i, data in enumerate(dataloader, 0):
    img, points_line, cat, points_square, fn, idx = data
    img = Variable(img)
    img = img.cuda()
    points_line = Variable(points_line)
    points_line = points_line.cuda()

    points_square = Variable(points_square)
    points_square = points_square.cuda()
    points = torch.cat((points_line, points_square), 1)

    pointsReconstructed_line = network_line.forward_inference(img, grid_line)
    pointsReconstructed_square = network_square.forward_inference(img, grid_square)
    pointsReconstructed = torch.cat((pointsReconstructed_line, pointsReconstructed_square), 1)
    pointsReconstructed_numpy = (pointsReconstructed.cpu().data).numpy()
    for j in range(img.size(0)):
        prediction = pointsReconstructed_numpy[j]
        print('Trainset: ', opt.trainset, 'Cat: ', catidx[cat[j]], 'Res: ', vx_res)
        if opt.trainset:
            outdir = '../Volume_refinement/data/%s/skeleton_prediction_binvox%d_train' % (catidx[cat[j]], vx_res)
        else:
            outdir = '../Volume_refinement/data/%s/skeleton_prediction_binvox%d_test' % (catidx[cat[j]], vx_res)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            print('creat dir', outdir)
        outfile = os.path.join(outdir, fn[j] + '_%02d.binvox' % int(idx[j]))
        if os.path.exists(outfile):
            print(outfile, 'have exists!!!')
            continue
        process_binvox(outfile, vx_res, prediction, fn[j], idx[j])
