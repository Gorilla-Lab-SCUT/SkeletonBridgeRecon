import os
import sys
import math
import numpy as np
import cPickle as pickle
from plyfile import PlyData,PlyElement

def normalize(vector, eps=1e-5):
    if len(vector.shape)==1:
        radius = np.linalg.norm(vector)
        return vector/(radius+eps)
    elif len(vector.shape)==2:
        radius = np.linalg.nor(vector, axis=1).max()
        return vector/(radius+eps)

sample_num = 10000
catname = '03001627'
data_root ='/data/tang.jiapeng/mesh_refinement_dataset/%s/%s_surface'%(catname,catname)
image_root ='/data/tang.jiapeng/ShapeNetAllRaw/ShapeNetRendering'
out_root ='/data/tang.jiapeng/mesh_refinement_dataset/%s/label'%catname
filetxt ='/data/tang.jiapeng/split_train_test/%s/all.txt'%catname
filelist=open(filetxt,'r').readlines()
filelist=[f.strip() for f in filelist]

if not os.path.exists(out_root):
	os.mkdir(out_root)

for f in filelist:
    plyfile=os.path.join(data_root,f+'_rec.ply')
    ptdata = PlyData.read(plyfile)['vertex'].data
    ptdata = ptdata.view(np.dtype('float32')).reshape(-1,6)
    if len(ptdata)<=sample_num:
        times = sample_num/len(ptdata)
        ptdata = np.repeat(ptdata,times,0)
        left = sample_num-len(ptdata)
        ptdata_left = ptdata[np.random.choice(ptdata.shape[0],left)]
        ptdata = np.concatenate((ptdata,ptdata_left),axis=0)
    else:
        ptdata = ptdata[np.random.choice(ptdata.shape[0],sample_num)]
    
    vertices = ptdata[:, 0:3]
    normals = ptdata[:, 3:6]
    
    params_path = os.path.join(image_root, catname, f, 'rendering','rendering_metadata.txt')
    params = open(params_path, 'r').readlines()
    for i in xrange(24):
        para = params[i]
        azimuth,elevation,_,distance,_=map(float,para.strip().split())
        ele = math.radians(elevation)
        azi= math.radians(azimuth)
        dis = distance
        eye = (dis * math.cos(ele) * math.cos(azi),
            dis * math.sin(ele),
            dis * math.cos(ele) * math.sin(azi))
        eye = np.asarray(eye)
        at = np.array([0, 0, 0],dtype='float32')
        up = np.array([0, 1, 0],dtype='float32')
        z_axis = normalize(eye - at, eps=1e-5) #forward
        x_axis = normalize(np.cross(up, z_axis), eps=1e-5) #left
        y_axis = normalize(np.cross(z_axis, x_axis), eps=1e-5) #up
        #rotation matrix: [3, 3]
        R = np.concatenate((x_axis[None, :], y_axis[None, :], z_axis[None, :]), axis=0)
        points = vertices - eye[None, :]
        points_rotate = points.dot(R.T)
        normals_rotate = normals.dot(R.T)

        ptdata = np.concatenate((points_rotate, normals_rotate),axis=1)
        outfile = os.path.join(out_root, f+'_%02d.pkl'%i)
        fout=open(outfile,'wb')
        pickle.dump(ptdata,fout,-1)
        print(outfile, ptdata.shape)
