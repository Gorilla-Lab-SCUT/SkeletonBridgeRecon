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
from model import *
from utils import *
from plyio import *
import torch.nn.functional as F
import sys
import os
import json
import time, datetime
import pandas as pd

sys.path.append("./extension/")
import dist_chamfer as ext

distChamfer = ext.chamferDist()

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--model_line', type=str, default='./trained_models/svr_line20_pretrained.pth',
                    help='your path to the trained model')
parser.add_argument('--model_square', type=str, default='./trained_models/svr_square20_pretrained.pth')
parser.add_argument('--num_points', type=int, default=2500, help='number of points fed to poitnet')
parser.add_argument('--gen_line_points', type=int, default=600, help='number of line points to generate')
parser.add_argument('--gen_square_points', type=int, default=2000, help='number of square points to generate')
parser.add_argument('--nb_primitives_line', type=int, default=20, help='number of primitives of squares')
parser.add_argument('--nb_primitives_square', type=int, default=20, help='number of primitives of squares')
parser.add_argument('--category', type=str, default='chair')
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNet(SVR=True, normal=False, class_choice=opt.category, train=False)  # train or test?
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False,
                                         num_workers=int(opt.workers))  # shuffle??
print('length of set ', len(dataset.datapath))
len_dataset = len(dataset)

cudnn.benchmark = True

network_line = AE_CurSkeNet(num_points=opt.gen_line_points, nb_primitives=opt.nb_primitives_line)
network_square = AE_SurSkeNet(num_points=opt.gen_square_points, nb_primitives=opt.nb_primitives_square)

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

grid = [vertices for i in range(0, opt.nb_primitives_square)]
grid2 = [vertices2 for i in range(0, opt.nb_primitives_line)]

# print("grain", grain, 'number vertices', len(vertices)*opt.nb_primitives)

results = dataset.cat.copy()
for i in results:
    results[i] = 0

# Iterate on the data
for i, data in enumerate(dataloader, 0):
    img, points_line, cat, points_square, fn, idx = data
    img = Variable(img)
    img = img.cuda()
    cat = cat[0]
    fn = fn[0]
    idx = idx[0]
    results[cat] = results[cat] + 1

    points_line = Variable(points_line, volatile=True)
    points_line_input = points_line.transpose(2, 1).contiguous()
    points_line_input = points_line_input.cuda()
    points_line = points_line.cuda()

    points_square = Variable(points_square, volatile=True)
    points_square_input = points_square.transpose(2, 1).contiguous()
    points_square_input = points_square_input.cuda()
    points_square = points_square.cuda()
    points = torch.cat((points_line, points_square), 1)

    pointsReconstructed_line = network_line.forward_inference(points_line_input, grid2)
    pointsReconstructed_square = network_square.forward_inference(points_square_input, grid)
    pointsReconstructed = torch.cat((pointsReconstructed_line, pointsReconstructed_square), 1)

    dist1, dist2 = distChamfer(points, pointsReconstructed)
    loss_net = ((torch.mean(dist1) + torch.mean(dist2)))

    print(results)
    outdir = './output/%s_AE_Curve%d_Sheet%d'%(opt.category, opt.gen_line_points, opt.gen_square_points)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print('creat dir', outdir)

    write_ply(filename=outdir + "/" + fn + "_" + idx + "_line_gt",
              points=pd.DataFrame((points_line.cpu().data.squeeze()).numpy()), as_text=True)
    write_ply(filename=outdir + "/" + fn + "_" + idx + "_line_gen",
              points=pd.DataFrame((pointsReconstructed_line.cpu().data.squeeze()).numpy()), as_text=True)
    write_ply(filename=outdir + "/" + fn + "_" + idx + "_square_gt",
              points=pd.DataFrame((points_square.cpu().data.squeeze()).numpy()), as_text=True)
    write_ply(filename=outdir + "/" + fn + "_" + idx + "_square_gen",
              points=pd.DataFrame((pointsReconstructed_square.cpu().data.squeeze()).numpy()), as_text=True)
    write_ply(filename=outdir + "/" + fn + "_" + idx,
              points=pd.DataFrame((pointsReconstructed.cpu().data.squeeze()).numpy()), as_text=True)
