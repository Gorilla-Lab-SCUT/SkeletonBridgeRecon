from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import sys
sys.path.append('./auxiliary/')
from dataset import *
from model import *
from utils import *
from nnutils import *
from plyio import *
import torch.nn.functional as F
import sys
import os
import json
import time, datetime
import visdom

sys.path.append("./extension/")
import dist_chamfer as ext

distChamfer = ext.chamferDist()

best_val_loss = 10

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--nepoch', type=int, default=80, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--num_points', type=int, default=2000, help='number of points')
parser.add_argument('--nb_primitives', type=int, default=20, help='number of primitives')
parser.add_argument('--super_points', type=int, default=2500,
                    help='number of input points to pointNet, not used by default')
parser.add_argument('--env', type=str, default="main", help='visdom env')
parser.add_argument('--k1', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_epoch', type=int, default=60)
parser.add_argument('--category', type=str, default='chair')
opt = parser.parse_args()
print(opt)

# Launch visdom for visualization
vis = visdom.Visdom(port=9000, env=opt.env)
now = datetime.datetime.now()
save_path = opt.env #now.isoformat() + 
dir_name = os.path.join('./log/%s_log' % opt.category, save_path)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
logname = os.path.join(dir_name, 'log.txt')

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Create train/test dataloader on new views and test dataset on new models
dataset = ShapeNet(normal=False, train=True, class_choice=opt.category, npoints_line=opt.super_points, npoints_square=opt.super_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

dataset_test = ShapeNet(normal=False, train=False, class_choice=opt.category, npoints_line=opt.super_points, npoints_square=opt.super_points)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))

print('training set', len(dataset.datapath))
print('testing set', len(dataset_test.datapath))

cudnn.benchmark = True
len_dataset = len(dataset)

#define intial squares
grid, faces_array, vertex_adj_matrix_tensor = define_squares(num_points=opt.num_points, nb_primitives=opt.nb_primitives)

# Create network
network = AE_SurSkeNet(num_points=opt.num_points, nb_primitives=opt.nb_primitives)
network.apply(weights_init)
network.cuda()
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")

print(network)

lrate = opt.lr
optimizer = optim.Adam(network.parameters(), lr=lrate)

num_batch = len(dataset) / opt.batchSize
train_loss = AverageValueMeter()
loss1 = AverageValueMeter()
loss2 = AverageValueMeter()
val_loss = AverageValueMeter()
val_loss1 = AverageValueMeter()
val_loss2 = AverageValueMeter()

with open(logname, 'a') as f:  # open and append
    f.write(str(network) + '\n')

train_curve = []
train_curve1 = []
train_curve2 = []
val_curve = []
val_curve1 = []
val_curve2 = []

trainloss_acc0 = 1e-9
trainloss_accs = 0
validloss_acc0 = 1e-9
validloss_accs = 0

for epoch in range(opt.nepoch):
    # TRAIN MODE
    train_loss.reset()
    loss1.reset()
    loss2.reset()
    network.train()

    if epoch == opt.lr_decay_epoch:
        optimizer = optim.Adam(network.parameters(), lr=lrate / 10.0)
        lrate = lrate / 10.0

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()

        _, _, cat, points, _, _ = data
        points = Variable(points)
        points_input = points.transpose(2, 1).contiguous()
        points_input = points_input.cuda()
        points = points.cuda()

        pointsReconstructed = network.forward_inference(points_input, grid)
        dist1, dist2 = distChamfer(points, pointsReconstructed)
        laplacian_smooth = surface_laplacian(pointsReconstructed, opt.nb_primitives, vertex_adj_matrix_tensor)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2)) + opt.k1 * (torch.mean(laplacian_smooth))
        trainloss_accs = trainloss_accs * 0.99 + loss_net.item()
        trainloss_acc0 = trainloss_acc0 * 0.99 + 1
        loss_net.backward()
        train_loss.update(loss_net.item())
        loss1.update(((torch.mean(dist1)) + (torch.mean(dist2))).item())
        loss2.update((torch.mean(laplacian_smooth)).item())

        optimizer.step()
        # VIZUALIZE
        if i % 50 <= 0:
            vis.scatter(X=points[0].data.cpu(),
                        win='TRAIN_INPUT',
                        opts=dict(
                            title="TRAIN_INPUT",
                            markersize=2,
                        ),
                        )
            vis.scatter(X=pointsReconstructed[0].data.cpu(),
                        win='TRAIN_INPUT_RECONSTRUCTED',
                        opts=dict(
                            title="TRAIN_INPUT_RECONSTRUCTED",
                            markersize=2,
                        ),
                        )

        print('[%d: %d/%d] train loss:  %f , %f ' % (
        epoch, i, len(dataset) / opt.batchSize, loss_net.item(), trainloss_accs / trainloss_acc0))
        print('CD loss : %f ; laplacian loss : %f' % (
        ((torch.mean(dist1)) + (torch.mean(dist2))).item(), (torch.mean(laplacian_smooth)).item()))

    # UPDATE CURVES
    train_curve.append(train_loss.avg)
    train_curve1.append(loss1.avg)
    train_curve2.append(loss2.avg)

    with torch.no_grad():
        # VALIDATION
        val_loss.reset()
        val_loss1.reset()
        val_loss2.reset()
        for item in dataset_test.cat:
            dataset_test.perCatValueMeter[item].reset()

        network.eval()
        for i, data in enumerate(dataloader_test, 0):
            _, _, cat, points, _, _ = data
            points = Variable(points)
            points_input = points.transpose(2, 1).contiguous()
            points_input = points_input.cuda()
            points = points.cuda()

            pointsReconstructed = network.forward_inference(points_input, grid)
            dist1, dist2 = distChamfer(points, pointsReconstructed)
            laplacian_smooth = surface_laplacian(pointsReconstructed, opt.nb_primitives, vertex_adj_matrix_tensor)
            loss_net = (torch.mean(dist1)) + (torch.mean(dist2)) + opt.k1 * (torch.mean(laplacian_smooth))
            validloss_accs = validloss_accs * 0.99 + loss_net.item()
            validloss_acc0 = validloss_acc0 * 0.99 + 1
            val_loss.update(loss_net.item())
            val_loss1.update((torch.mean(dist1) + torch.mean(dist2)).item())
            val_loss2.update((torch.mean(laplacian_smooth)).item())
            dataset_test.perCatValueMeter[cat[0]].update((torch.mean(dist1) + torch.mean(dist2)).item())
            if i % 25 == 0:
                vis.scatter(X=points[0].data.cpu(),
                            win='VAL_INPUT',
                            opts=dict(
                                title="VAL_INPUT",
                                markersize=2,
                            ),
                            )
                vis.scatter(X=pointsReconstructed[0].data.cpu(),
                            win='VAL_INPUT_RECONSTRUCTED',
                            opts=dict(
                                title="VAL_INPUT_RECONSTRUCTED",
                                markersize=2,
                            ),
                            )
            print('[%d: %d/%d] valid loss:  %f , %f ' % (
            epoch, i, len(dataset_test) / opt.batchSize, loss_net.item(), validloss_accs / validloss_acc0))
            print('CD loss : %f ; laplacian loss : %f' % (
            ((torch.mean(dist1)) + (torch.mean(dist2))).item(), (torch.mean(laplacian_smooth)).item()))

        # UPDATE CURVES
        val_curve.append(val_loss.avg)
        val_curve1.append(val_loss1.avg)
        val_curve2.append(val_loss2.avg)

    vis.line(
        X=np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve)))),
        Y=np.log(np.column_stack((np.array(train_curve), np.array(val_curve)))),
        win='All loss log',
        opts=dict(title="All loss", legend=["train_curve" + opt.env, "val_curve" + opt.env],
                  markersize=2, ), )
    vis.line(
        X=np.column_stack((np.arange(len(train_curve1)), np.arange(len(val_curve1)))),
        Y=np.log(np.column_stack((np.array(train_curve1), np.array(val_curve1)))),
        win='CD loss log',
        opts=dict(title="CD loss", legend=["train_curve1" + opt.env, "val_curve1" + opt.env],
                  markersize=2, ), )
    vis.line(
        X=np.column_stack((np.arange(len(train_curve2)), np.arange(len(val_curve2)))),
        Y=np.log(np.column_stack((np.array(train_curve2), np.array(val_curve2)))),
        win='Laplacian loss log',
        opts=dict(title="Laplacian loss", legend=["train_curve2" + opt.env, "val_curve2" + opt.env],
                  markersize=2, ), )
    log_table = {
        "train_loss": train_loss.avg,
        "val_loss": val_loss.avg,
        "epoch": epoch,
        "lr": lrate,
        "super_points": opt.super_points,
        "bestval": best_val_loss,
    }
    print(log_table)
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    with open(logname, 'a') as f:  # open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
    if best_val_loss > val_loss.avg:
        best_val_loss = val_loss.avg
        print('New best loss : ', best_val_loss)
        print('saving net...')
        torch.save(network.state_dict(), '%s/network.pth' % dir_name)