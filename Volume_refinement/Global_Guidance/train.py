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

best_val_loss = 100
best_val3_loss = 100

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--env', type=str, default="Global_Guidance", help='visdom env')
parser.add_argument('--nepoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--lr',type=float,default=1e-3, help='Initial learning rate.')
parser.add_argument('--lrDecay',type=float, default=0.1)
parser.add_argument('--lrStep',type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=3e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--category', type=str, default='chair')
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
dataset = Global_ShapeNet(cat=class_name[opt.category], train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_test = Global_ShapeNet(cat=class_name[opt.category], train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
print('training set', len(dataset))
print('testing set', len(dataset_test))

cudnn.benchmark = True
len_dataset = len(dataset)
len_dataset_test = len(dataset_test)

# Create network
network = Global_Guidance()
network.apply(weights_init)
network.cuda()
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")

# Create Loss Module
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()

# Create optimizer
lrate = opt.lr
optimizer = optim.Adam(network.parameters(), lr=lrate, weight_decay=opt.weight_decay)

with open(logname, 'a') as f:  # open and append
    f.write(str(network) + '\n')

train_curve = []
train_curve1 =[]
train_curve2 = []
train_curve3 = []
val_curve = []
val_curve1 = []
val_curve2 = []
val_curve3 = []

train_loss = AverageValueMeter()
train_loss1 = AverageValueMeter()
train_loss2 = AverageValueMeter()
train_loss3 = AverageValueMeter()
val_loss = AverageValueMeter()
val_loss1 = AverageValueMeter()
val_loss2 = AverageValueMeter()
val_loss3 = AverageValueMeter()
trainloss_acc0 = 1e-9
trainloss_accs = 0
validloss_acc0 = 1e-9
validloss_accs = 0

for epoch in range(opt.nepoch):
    # TRAIN MODE
    train_loss.reset()
    train_loss1.reset()
    train_loss2.reset()
    train_loss3.reset()
    network.train()
    if epoch !=0 and epoch % opt.lrStep == 0 :
        #lrate = opt.lr * opt.lrDecay ** (epoch//opt.lrStep)
        lrate = lrate * opt.lrDecay
        optimizer = optim.Adam(network.parameters(), lr= lrate, weight_decay=opt.weight_decay)
        print('learning rate decay', lrate)
    for i, data in enumerate(dataloader, 0):
        t0 = time.time()
        optimizer.zero_grad()
        img, input32, gt32, input64, gt64, f, mod, seq = data
        img = img.cuda()
        input32 = input32.cuda()
        gt32 = gt32.cuda()
        input64 = input64.cuda()
        gt64 = gt64.cuda()

        prediction1, prediction2, prediction3 = network(img, input32, input64)
        loss1 = criterion1(prediction1, gt32)
        loss2 = criterion1(prediction2, gt32)
        loss3 = criterion2(prediction3, gt64)
        loss_net = loss1 + loss2 + loss3
        trainloss_accs = trainloss_accs * 0.99 + loss_net.item()
        trainloss_acc0 = trainloss_acc0 * 0.99 + 1
        loss_net.backward()
        train_loss.update(loss_net.item())
        train_loss1.update(loss1.item())
        train_loss2.update(loss2.item())
        train_loss3.update(loss3.item())
        optimizer.step()
        print('[%d: %d/%d] train loss:  %f , %f , %f , %f,  time: %f ' % (
            epoch, i, len_dataset / opt.batchSize, loss1.item(), loss2.item(), loss3.item(),
            trainloss_accs / trainloss_acc0, time.time() - t0))

    #UPDATE CURVES
    train_curve.append(train_loss.avg)
    train_curve1.append(train_loss1.avg)
    train_curve2.append(train_loss2.avg)
    train_curve3.append(train_loss3.avg)

    #VALIDATION
    val_loss.reset()
    val_loss1.reset()
    val_loss2.reset()
    val_loss3.reset()
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
            loss1 = criterion1(prediction1, gt32)
            loss2 = criterion1(prediction2, gt32)
            loss3 = criterion2(prediction3, gt64)
            loss_net = loss1 + loss2 + loss3
            validloss_accs = validloss_accs * 0.99 + loss_net.item()
            validloss_acc0 = validloss_acc0 * 0.99 + 1
            val_loss.update(loss_net.item())
            val_loss1.update(loss1.item())
            val_loss2.update(loss2.item())
            val_loss3.update(loss3.item())
            print('[%d: %d/%d] val loss:  %f , %f , %f , %f , time : %f ' % (
                epoch, i, len(dataset_test)/opt.batchSize, loss1.item(), loss2.item(), loss3.item(),
                validloss_accs / validloss_acc0, time.time()-t0))

        #UPDATE CURVES
        val_curve.append(val_loss.avg)
        val_curve1.append(val_loss1.avg)
        val_curve2.append(val_loss2.avg)
        val_curve3.append(val_loss3.avg)

    vis.line(X=np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve)))),
                 Y=np.column_stack((np.array(train_curve),np.array(val_curve))),
                 win='loss_all',
                 opts=dict(title="loss_all", legend=["train_curve", "val_curve"], markersize=2, ), )
    
    vis.line(X=np.column_stack((np.arange(len(train_curve1)), np.arange(len(val_curve1)))),
                 Y=np.column_stack((np.array(train_curve1),np.array(val_curve1))),
                 win='loss_ce1',
                 opts=dict(title="loss_ce1", legend=["train_curve_ae32", "val_curve_ae32"], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_curve2)), np.arange(len(val_curve2)))),
                 Y=np.column_stack((np.array(train_curve2),np.array(val_curve2))),
                 win='loss_ce2',
                 opts=dict(title="loss_ce2", legend=["train_curve_ref32", "val_curve_ref32"], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_curve3)), np.arange(len(val_curve3)))),
                 Y=np.column_stack((np.array(train_curve3),np.array(val_curve3))),
                 win='loss_ce3',
                 opts=dict(title="loss_ce3", legend=["train_curve_ref64", "val_curve_ref64"], markersize=2, ), )
    #show log of loss
    vis.line(X=np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve)))),
                 Y=np.log(np.column_stack((np.array(train_curve),np.array(val_curve)))),
                 win='log_all',
                 opts=dict(title="log_all", legend=["train_curve", "val_curve"], markersize=2, ), )
    vis.line(X=np.column_stack((np.arange(len(train_curve3)), np.arange(len(val_curve3)))),
                 Y=np.column_stack((np.array(train_curve3),np.array(val_curve3))),
                 win='log_ce3',
                 opts=dict(title="log_ce3", legend=["train_curve_ref64", "val_curve_ref64"], markersize=2, ), )
    log_table = {
      "train_loss" : train_loss.avg,
      "val_loss" : val_loss.avg,
      "epoch" : epoch,
      "lr" : lrate,
      "bestval" : best_val_loss,
      "bestval_min64" : best_val3_loss,
    }
    print(log_table)

    with open(logname, 'a') as f:
        f.write('json_stats: ' + json.dumps(log_table) + '\n')

    if best_val_loss > val_loss.avg:
        best_val_loss = val_loss.avg
        print('New best valid loss : ', best_val_loss)
        print('saving net...')
        #torch.save(network.state_dict(), '%s/network_epoch%d.pth' % (dir_name, epoch))
        torch.save(network.state_dict(), '%s/network.pth' % dir_name)

    if best_val3_loss > val_loss3.avg:
        best_val3_loss = val_loss3.avg
        print('New best valid3 loss : ', best_val3_loss)
        print('saving net...')
        #torch.save(network.state_dict(), '%s/network_min64_epoch%d.pth' % (dir_name, epoch))
        torch.save(network.state_dict(), '%s/network_min64.pth' % dir_name)
