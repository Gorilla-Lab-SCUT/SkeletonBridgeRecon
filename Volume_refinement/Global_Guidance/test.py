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
from tqdm import tqdm
import os
import json
import time, datetime
import visdom
sys.path.append('./auxiliary/')
from dataset_global import *
from layers import *
from model_global import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--th', type=float, default=0.4, help='the thresold to compute IoU')
parser.add_argument('--category', type=str, default='chair')
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset_test = Global_ShapeNet(cat=class_name[opt.category], train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
print('testing set', len(dataset_test))
len_dataset_test = len(dataset_test)
# Create network
network = Global_Guidance()
network.apply(weights_init)
network.cuda()
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
else:
    print(" Give your trained model")
    exit()

# Create Loss Module
criterion = nn.CrossEntropyLoss()

iou1 = AverageValueMeter()
iou2 = AverageValueMeter()
iou3 = AverageValueMeter()
pre = AverageValueMeter()
rec = AverageValueMeter()
iou1.reset()
iou2.reset()
iou3.reset()
pre.reset()
rec.reset()
network.eval()
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        t0 = time.time()
        img, input32, gt32, input64, gt64, f, mod, seq = data
        img = img.cuda()
        input32 = input32.cuda()
        gt32 = gt32.cuda()
        input64 = input64.cuda()
        gt64 = gt64.cuda()

        prediction1, prediction2, prediction3 = network(img, input32, input64)
        cross_entropy = criterion(prediction3, gt64)
        prediction1 = F.softmax(prediction1, dim=1)
        prediction1 = torch.ge(prediction1[:, 1, :, :, :], opt.th)
        prediction1 = prediction1.type(torch.cuda.FloatTensor)
        gt32 = gt32.type(torch.cuda.FloatTensor)
        inter1 = torch.min(prediction1, gt32).sum(3).sum(2).sum(1)
        union1 = torch.max(prediction1, gt32).sum(3).sum(2).sum(1)
        inter_over_union1 = torch.mean(inter1 / union1)
        iou1.update(inter_over_union1.item())

        prediction2 = F.softmax(prediction2, dim=1)
        prediction2 = torch.ge(prediction2[:, 1, :, :, :], opt.th)
        prediction2 = prediction2.type(torch.cuda.FloatTensor)
        inter2 = torch.min(prediction2, gt32).sum(3).sum(2).sum(1)
        union2 = torch.max(prediction2, gt32).sum(3).sum(2).sum(1)
        inter_over_union2 = torch.mean(inter2 / union2)
        iou2.update(inter_over_union2.item())

        prediction3 = F.softmax(prediction3, dim=1)
        prediction3 = torch.ge(prediction3[:, 1, :, :, :], opt.th)
        prediction3 = prediction3.type(torch.cuda.FloatTensor)
        gt64 = gt64.type(torch.cuda.FloatTensor)
        inter3 = torch.min(prediction3, gt64).sum(3).sum(2).sum(1)
        union3 = torch.max(prediction3, gt64).sum(3).sum(2).sum(1)
        prediction_sum = prediction3.sum(3).sum(2).sum(1)
        gt_sum = gt64.sum(3).sum(2).sum(1)

        inter_over_union3 = torch.mean(inter3 / union3)
        precision = torch.mean(inter3 / (prediction_sum + 1e-6))
        recall = torch.mean(inter3 / gt_sum)
        iou3.update(inter_over_union3.item())
        pre.update(precision.item())
        rec.update(recall.item())
        print('[%d/%d] mean iou img32: %f , mean iou input32: %f , mean iou input64: %f, precision: %f, recall: %f, time: %f' %
            (i, len(dataset_test) / opt.batchSize, iou1.avg, iou2.avg, iou3.avg, pre.avg, rec.avg, time.time() - t0))
