from __future__ import print_function
import argparse
import os
import random
import numpy as np
import math
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
import subprocess
import pandas as pd
from PIL import Image
import scipy.io as sio

sys.path.append("./extension/")
import dist_chamfer as ext

distChamfer = ext.chamferDist()


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--model_line', type=str, default='./trained_models/svr_line20_pretrained.pth',
                    help='your path to the trained model')
parser.add_argument('--model_square', type=str, default='./trained_models/svr_square20_pretrained.pth')
parser.add_argument('--num_points', type=int, default=2600, help='number of points fed to poitnet')
parser.add_argument('--gen_line_points', type=int, default=600, help='number of line points to generate')
parser.add_argument('--gen_square_points', type=int, default=2000, help='number of square points to generate')
parser.add_argument('--nb_primitives_line', type=int, default=20, help='number of primitives of squares')
parser.add_argument('--nb_primitives_square', type=int, default=20, help='number of primitives of squares')
parser.add_argument('--category', type=str, default='chair')
parser.add_argument('--modname', type=str, default='1006be65e7bc937e9141f9b58470d646')
parser.add_argument('--image_path', type=str, default='1006be65e7bc937e9141f9b58470d646/rendering/00.png',
                    help='the path of test image')

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

catfile = os.path.join('./data/synsetoffset2category.txt')
cat = {}
with open(catfile, 'r') as f:
    for line in f:
        ls = line.strip().split()
        cat[ls[0]] = ls[1]
print(cat)


def load_image_point(catname, modname):
    batch_data = []
    batch_line = []
    batch_square = []
    batch_mod = []
    batch_index = []
    for i in range(24):
        image = os.path.join('./data/ShapeNetRendering', catname, modname, 'rendering',
                             '%02d.png' % i)
        im = Image.open(image)
        center_crop = transforms.Compose([transforms.CenterCrop(127), ])
        scale = transforms.Compose([transforms.Resize(size=224, interpolation=2), transforms.ToTensor()])
        im = center_crop(im)  # center crop
        data = scale(im)  # scale
        data = data[:3, :, :]

        line_path = os.path.join('./data/ShapeNetPointCloud', catname, 'classify_mat',
                                 modname + '_line.mat')
        square_path = os.path.join('./data/ShapeNetPointCloud', catname, 'classify_mat',
                                   modname + '_square.mat')
        point_set_line = sio.loadmat(line_path)['v']
        point_set_square = sio.loadmat(square_path)['v']

        para_path = os.path.join('./data/ShapeNetRendering', catname, modname, "rendering",
                                 "rendering_metadata.txt")
        params = open(para_path).readlines()[i]
        azimuth, elevation, _, distance, _ = map(float, params.strip().split())
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

        # point_set_line = point_set_line - eye[None, :]
        # point_set_square = point_set_square - eye[None, :]
        point_set_line = point_set_line.dot(R.T)
        point_set_square = point_set_square.dot(R.T)
        if len(point_set_line) <= 2500:
            times = 5000 / len(point_set_line)
            point_set_line = np.repeat(point_set_line, times, 0)
            left = 5000 - len(point_set_line)
            point_left = point_set_line[np.random.choice(point_set_line.shape[0], left)]
            point_set_line = np.concatenate((point_set_line, point_left), axis=0)
        else:
            point_set_line = point_set_line[np.random.choice(point_set_line.shape[0], 5000)]
        if len(point_set_square) <= 5000:
            times = 5000 / len(point_set_square)
            point_set_square = np.repeat(point_set_square, times, 0)
            left = 5000 - len(point_set_square)
            point_left = point_set_square[np.random.choice(point_set_square.shape[0], left)]
            point_set_square = np.concatenate((point_set_square, point_left), axis=0)
        else:
            point_set_square = point_set_square[np.random.choice(point_set_square.shape[0], 5000)]
            point_set_line = torch.from_numpy(point_set_line).type(torch.FloatTensor)
            point_set_square = torch.from_numpy(point_set_square).type(torch.FloatTensor)

            data = torch.unsqueeze(data, 0).contiguous()
            point_set_line = torch.unsqueeze(point_set_line, 0).contiguous()
            point_set_square = torch.unsqueeze(point_set_square, 0).contiguous()
        batch_data.append(data)
        batch_line.append(point_set_line)
        batch_square.append(point_set_square)
        batch_mod.append(modname)
        batch_index.append('%02d' % i)

    mybatch_data = torch.cat(batch_data, dim=0)
    mybatch_line = torch.cat(batch_line, dim=0)
    mybatch_square = torch.cat(batch_square, dim=0)
    return mybatch_data, mybatch_line, mybatch_square, batch_mod, batch_index


cudnn.benchmark = True
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

outroot = './output/demo_%s' % opt.category
if not os.path.exists(outroot):
    os.makedirs(outroot)
    print('creat root', outroot)

if True:
    img, points_line, points_square, fn, idx = load_image_point(cat[opt.category], opt.modname)
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
    outdir = os.path.join(outroot, opt.modname)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        print('creat dir', outdir)
    for j in range(img.size(0)):
        write_ply(filename=outdir + "/" + fn[j] + "_" + idx[j] + "_line_gt",
                  points=pd.DataFrame((points_line[j].cpu().data.squeeze()).numpy()), as_text=True)
        write_ply(filename=outdir + "/" + fn[j] + "_" + idx[j] + "_line_gen",
                  points=pd.DataFrame((pointsReconstructed_line[j].cpu().data.squeeze()).numpy()), as_text=True)
        write_ply(filename=outdir + "/" + fn[j] + "_" + idx[j] + "_square_gt",
                  points=pd.DataFrame((points_square[j].cpu().data.squeeze()).numpy()), as_text=True)
        write_ply(filename=outdir + "/" + fn[j] + "_" + idx[j] + "_square_gen",
                  points=pd.DataFrame((pointsReconstructed_square[j].cpu().data.squeeze()).numpy()), as_text=True)
        write_ply(filename=outdir + "/" + fn[j] + "_" + idx[j],
                  points=pd.DataFrame((pointsReconstructed[j].cpu().data.squeeze()).numpy()), as_text=True)
    print("Successfully saved to : %s"%outdir)
