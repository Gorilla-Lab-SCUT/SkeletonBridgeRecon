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
from layers import *
from dataset_local import *
from model_local import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
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

dataset_test = Local_ShapeNet(cat=class_name[opt.category], train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
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

# Create Loss Module
criterion = nn.CrossEntropyLoss()
iou = AverageValueMeter()
pre = AverageValueMeter()
rec = AverageValueMeter()
iou.reset()
pre.reset()
rec.reset()
network.eval()
with torch.no_grad():
    for idx in xrange(len_dataset_test / 8):
        t0 = time.time()
        refine, input, gt, f, mod, seq = dataset_test.get_batch()
        refine = refine.cuda()
        input = input.cuda()
        gt = gt.cuda()

        output = network(refine, input)
        output = F.softmax(output, dim=1)
        output = torch.ge(output[:, 1, :, :, :], opt.th).type(torch.cuda.FloatTensor)
        gt = gt.type(torch.cuda.FloatTensor)

        prediction = torch.cuda.FloatTensor(1, 128, 128, 128)
        groudtruth = torch.cuda.FloatTensor(1, 128, 128, 128)
        for i in xrange(2):
            for j in xrange(2):
                for k in xrange(2):
                    prediction[:, i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, k * 64:(k + 1) * 64] = output.data[i * 4 + j * 2 + k, :, :, :]
                    groudtruth[:, i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, k * 64:(k + 1) * 64] = gt.data[i * 4 + j * 2 + k, :, :, :]
        inter = torch.min(prediction, groudtruth).sum(3).sum(2).sum(1)
        union = torch.max(prediction, groudtruth).sum(3).sum(2).sum(1)
        prediction_sum = prediction.sum(3).sum(2).sum(1)
        gt_sum = groudtruth.sum(3).sum(2).sum(1)

        inter_over_union = torch.mean(inter / union)
        precision = torch.mean(inter / (prediction_sum + 1e-6))
        recall = torch.mean(inter / gt_sum)
        iou.update(inter_over_union.item())
        pre.update(precision.item())
        rec.update(recall.item())
        print('[%d/%d] mean iou: %f , precision: %f, recall: %f, time: %f' %
            (idx, len(dataset_test) / 8, iou.avg, pre.avg, rec.avg, time.time() - t0))