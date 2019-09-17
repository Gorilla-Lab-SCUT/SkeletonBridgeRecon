import os
import sys
import ctypes as ct
import numpy as np
import time
import random
import multiprocessing
import cPickle as pickle
import scipy.misc
import scipy.io
from plyfile import PlyData,PlyElement,make2d
import trimesh

n_processes = 10
suffix = '.obj'
catname = '03001627'
basemesh_dir = '/data/tang.jiapeng/mesh_refinement_dataset/%s/baseMesh_simplify'%catname
output_dir = '/data/tang.jiapeng/mesh_refinement_dataset/%s/basemesh_data'%catname
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

fns = []
for f in sorted(os.listdir(basemesh_dir)):
    fns.append(f)
print len(fns)

def worker(filename):
    t = time.time()
    objfile = os.path.join(basemesh_dir, filename)
    pklfile = os.path.join(output_dir, filename[:-4]+'.pkl')
    if os.path.exists(pklfile):
        print(pklfile,'has exists!!!')
    else:
        mesh = trimesh.load_mesh(objfile)
        #vertices
        vertices = mesh.vertices
        vertices = vertices.astype('float32')
        #faces
        faces = mesh.faces
        faces = faces.astype('int32')
        #halfedges
        halfedges = mesh.edges
        halfedges = halfedges.astype('int32')

        fout = open(pklfile,'wb')
        pickle.dump((vertices,halfedges,faces),fout,-1)
        print time.time()-t, filename[:-4]

pool = multiprocessing.Pool(processes=n_processes)
for idx,f in enumerate(fns):
	if n_processes >1:
		pool.apply_async(worker,args=(f,))
	else:
		worker(f)

if n_processes >1:
	pool.close()
	pool.join()
