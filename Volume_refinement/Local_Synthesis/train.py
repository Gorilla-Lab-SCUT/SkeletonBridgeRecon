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

best_val_loss = 10

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--env', type=str, default="Local_Synthesis", help='visdom env')
parser.add_argument('--nepoch', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--lr',type=float,default=1e-4, help='Initial learning rate.')
parser.add_argument('--lrDecay',type=float, default=0.1)
parser.add_argument('--lrStep',type=int, default=2)
parser.add_argument('--weight_decay', type=float, default=3e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--category', type=str, default='chair')
parser.add_argument('--iters',type=int, default=10000)
opt = parser.parse_args()
print(opt)

# Launch visdom for visualization
vis = visdom.Visdom(port=9000, env=opt.env)
now = datetime.datetime.now()
save_path = now.isoformat() + opt.env #now.isoformat()
dir_name = os.path.join('./log/%s_log' % opt.category, save_path)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
logname = os.path.join(dir_name, 'log.txt')

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Creat train/test dataloader
dataset = Local_ShapeNet(cat=class_name[opt.category], train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_test = Local_ShapeNet(cat=class_name[opt.category], train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
print('training set', len(dataset))
print('testing set', len(dataset_test))

cudnn.benchmark = True
len_dataset = len(dataset)
len_dataset_test = len(dataset_test)

# Create network
network = Local_Synthesis()
network.apply(weights_init)
network.cuda()
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")

# Create Loss Module
criterion = nn.CrossEntropyLoss()

# Create optimizer
lrate = opt.lr
optimizer = optim.Adam(network.parameters(), lr=lrate, weight_decay=opt.weight_decay)

with open(logname, 'a') as f:  # open and append
    f.write(str(network) + '\n')

train_curve = []
val_curve = []
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()

trainloss_acc0 = 1e-9
trainloss_accs = 0
validloss_acc0 = 1e-9
validloss_accs = 0

for epoch in range(opt.nepoch):
    # TRAIN MODE
    train_loss.reset()
    network.train()
    if epoch !=0 and epoch % opt.lrStep == 0 :
        #lrate = opt.lr * opt.lrDecay ** (epoch//opt.lrStep)
        lrate = lrate * opt.lrDecay
        optimizer = optim.Adam(network.parameters(), lr= lrate, weight_decay=opt.weight_decay)
        print('learning rate decay', lrate)
    for idx, data in enumerate(dataloader, 0):
        t0 = time.time()
        optimizer.zero_grad()
        refine, input, gt, i, j, k, f, mod, seq = data
        #refine, input, gt, f, mod, seq = dataset.get_batch()
        refine = refine.cuda()
        input = input.cuda()
        gt = gt.cuda()

        prediction = network(refine, input)
        loss_net = criterion(prediction, gt)
        trainloss_accs = trainloss_accs * 0.99 + loss_net.item()
        trainloss_acc0 = trainloss_acc0 * 0.99 + 1
        loss_net.backward()
        train_loss.update(loss_net.item())
        optimizer.step()

        print('[%d: %d/%d] train loss:  %f , %f , time: %f' % (
            epoch, idx, len_dataset / opt.batchSize, loss_net.item(), trainloss_accs / trainloss_acc0, time.time() - t0))
        if idx % opt.iters == 0:
            train_curve.append(trainloss_accs / trainloss_acc0)
            vis.line(X=np.arange(len(train_curve)), Y=np.array(train_curve), win='loss_train', opts=dict(title="loss_train", legend=["train_curve"], markersize=2, ), )
            vis.line(X=np.arange(len(train_curve)), Y=np.log(np.array(train_curve)), win='loss_train', opts=dict(title="loss_train_log", legend=["train_curve_log"], markersize=2, ), )
            print('saving net...')
            #torch.save(network.state_dict(), '%s/network_epoch%d_iter%d.pth' % (dir_name, epoch, idx))
            torch.save(network.state_dict(), '%s/network.pth' % dir_name)
    print('saving net...')
    torch.save(network.state_dict(), '%s/network.pth' % dir_name)
