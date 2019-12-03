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
from tqdm import tqdm
import json
import time, datetime
from PIL import Image
import sys
sys.path.append('./Skeleton_inference/auxiliary')
from model import *
from plyio import *
from binvox_rw import *
sys.path.append('./Volume_refinement/auxiliary')
from model_global import *
from model_local import *
sys.path.append('./Volume_refinement')
import external.libmcubes as libmcubes  #from external import libmcubes
import external.libsimplify as libsimplify #from external import libsimplify
import trimesh

def load_image(image_path):
    im = Image.open(image_path)
    crop = transforms.Compose([
            transforms.CenterCrop(127),
        ])
    resize = transforms.Compose([
            transforms.Scale(size=224, interpolation=2),
            transforms.ToTensor(),
        ])
    #scale = transforms.Compose([transforms.Scale(size =  224, interpolation = 2), 
        #transforms.Compose([transforms.CenterCrop(224),]), 
        #transforms.ToTensor()])
    #data = scale(im)
    data = resize(crop(im))
    data = data[None, :3,:,:]
    return data


parser = argparse.ArgumentParser()
parser.add_argument('--gen_line_points', type=int, default = 2400,  help='number of line points to generate')
parser.add_argument('--gen_square_points', type=int, default = 32000,  help='number of square points to generate')
parser.add_argument('--nb_primitives_line', type=int, default = 20,  help='number of primitives of squares')
parser.add_argument('--nb_primitives_square', type=int, default = 20,  help='number of primitives of squares')
parser.add_argument('--nfaces', type=int, default = 10000,  help='number of faces of base meshes')
parser.add_argument('--category', type=str, default='chair')
parser.add_argument('--th',type=float, default = 0.4)
opt = parser.parse_args()
print (opt)

model_line = 'Skeleton_inference/trained_models/%s/svr_cur/network.pth'%opt.category
model_square = 'Skeleton_inference/trained_models/%s/svr_sur/network.pth'%opt.category
model_global = 'Volume_refinement/trained_models/%s/global.pth'%opt.category
model_local = 'Volume_refinement/trained_models/%s/local.pth'%opt.category

network_line = SVR_CurSkeNet(num_points = opt.gen_line_points, nb_primitives = opt.nb_primitives_line)
network_square = SVR_SurSkeNet(num_points=opt.gen_square_points,nb_primitives=opt.nb_primitives_square)
network_line.cuda()
network_square.cuda()
if model_line != '' and model_square!= '':
    network_line.load_state_dict(torch.load(model_line))
    network_square.load_state_dict(torch.load(model_square))
    print('Succefullly load the CurSkeNet and SurSkeNet model!!!')
else: 
    print('Please load the CurSkeNet and SurSkeNet model!!!')
network_line.eval()
network_square.eval()

grain = int(np.sqrt(opt.gen_square_points/opt.nb_primitives_square))-1
grain = grain*1.0
grain2 = int(opt.gen_line_points/opt.nb_primitives_line)-1
grain2 = grain2*1.0

vertices = []
vertices2 = []
for i in range(0,int(grain + 1 )):
        for j in range(0,int(grain + 1 )):
            vertices.append([i/grain,j/grain])
for i in range(0,int(grain2+1)):
    vertices2.append([i/grain2,0])
grid = [vertices for i in range(0,opt.nb_primitives_square)]
grid2=[vertices2 for i in range(0,opt.nb_primitives_line)]

global_guidance = Global_Guidance()
local_synthesis = Local_Synthesis()
global_guidance.cuda()
local_synthesis.cuda()
if model_line != '' and model_square!= '':
    global_guidance.load_state_dict(torch.load(model_global))
    local_synthesis.load_state_dict(torch.load(model_local))
    print('Succefullly load the Volume_refinement model!!!')
else:
    print('Please load the Volume_refinement model!!!')
global_guidance.eval()
local_synthesis.eval()

data_root = './demo'
img_dir = os.path.join(data_root, opt.category)
skedir = os.path.join(data_root, '%s_skeleton'%opt.category)
if not os.path.exists(skedir):
    os.mkdir(skedir)
meshdir = os.path.join(data_root,'%s_basemesh'%opt.category)
if not os.path.exists(meshdir):
    os.mkdir(meshdir)

fns = sorted(os.listdir(img_dir))
for file in fns:
    img_path = os.path.join(img_dir, file)
    print(img_path)
    img = load_image(img_path)
    img = img.cuda()

    #Stage one : skeleton inference
    CurveSkeleton = network_line.forward_inference(img, grid2)
    SheetSkeleton = network_square.forward_inference(img,grid)
    Skeleton = torch.cat((CurveSkeleton,SheetSkeleton),1)
    plyfile = os.path.join(skedir, file[:-4]+'.ply')
    write_ply(filename=plyfile, points=pd.DataFrame((Skeleton.cpu().data.squeeze()).numpy()), as_text=True)
    print('Skeleton Inference Finish!!!')

    #Stage Two: convert pointcloud to solid volume
    prediction = (Skeleton.cpu().data.squeeze()).numpy()
    MIN = np.min(prediction, 0)
    MAX = np.max(prediction, 0)
    translate = (MIN + MAX)*0.5
    translate = [float(x) for x in translate]
    scale = np.max(MAX - MIN)

    input32 = (prediction - translate)/scale*32 + (32-1.0)/2.0
    sizes = [32, 32, 32]
    input32 = sparse_to_dense(input32.T, sizes)#.astype('int32')
    input32 = input32.astype('float32')
    input32 = torch.from_numpy(input32[None, None, :, :, :]).type(torch.FloatTensor)
    input32 = input32.cuda()

    input64 = (prediction - translate)/scale*64 + (64-1.0)/2.0
    sizes = [64, 64, 64]
    input64 = sparse_to_dense(input64.T, sizes)#.astype('int32')
    input64 = input64.astype('float32')
    input64 = torch.from_numpy(input64[None, None, :, :, :]).type(torch.FloatTensor)
    input64 = input64.cuda()

    _, _, global_refine = global_guidance(img, input32, input64)
    global_refine = F.softmax(global_refine)
    global_refine = torch.ge(global_refine[:,1,:, :, :], opt.th)
    global_refine = global_refine.cpu().data.squeeze().numpy()
    print('Global Guidance Finish!!!', global_refine.shape)

    input128 = (prediction - translate)/scale*128 + (128-1.0)/2.0
    sizes = [128, 128, 128]
    input128 = sparse_to_dense(input128.T, sizes)#.astype('int32')
    input128 = input128.astype('float32')

    refine_batch = []
    input_batch = []
    for i in xrange(2):
        for j in xrange(2):
            for k in xrange(2):
                refine_batch.append(global_refine[None, None, i*32:(i+1)*32, j*32:(j+1)*32, k*32:(k+1)*32])
                input_batch.append(input128[None, None, i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64])
    refine_batch = np.concatenate(refine_batch, axis=0)
    input_batch = np.concatenate(input_batch, axis=0)
    refine_batch = torch.from_numpy(refine_batch).type(torch.FloatTensor)
    input_batch = torch.from_numpy(input_batch).type(torch.FloatTensor)
    refine_batch = refine_batch.cuda()
    input_batch = input_batch.cuda()

    output = local_synthesis(refine_batch, input_batch)
    output = F.softmax(output)
    output = torch.ge(output[:,1,:,:,:], opt.th)
    print('Local Synthesis Finish!!!', output.size())

    final = torch.cuda.FloatTensor(1, 128, 128, 128)
    for i in xrange(2):
        for j in xrange(2):
            for k in xrange(2):
                final[:, i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64] = output.data[i*4+j*2+k, :, :, :]
    objfile = os.path.join(meshdir, file[:-4]+'.obj')
    voxels = torch.squeeze(final).cpu().numpy()
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))  ## padding for generating a closed shape
    vertices, faces, = libmcubes.marching_cubes(voxels, 0.5)
    vertices = (vertices - 63.5)/128.0 * scale + np.array([0, 0, -0.8*1.75], dtype='f4')
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=None, process=False)
    mesh = libsimplify.simplify_mesh(mesh, opt.nfaces)
    mesh.export(objfile)
    print('Base Mesh Generation Finish!!!')
