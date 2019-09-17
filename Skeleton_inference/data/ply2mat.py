import os
import numpy as np
import scipy.io as sio
from plyfile import PlyData,PlyElement

cats = [
'02691156',
'02828884',
'02933112',
'02958343',
'03001627',
'03211117',
'03636649',
'03691459',
'04090263',
'04256520',
'04379243',
'04401088',
'04530566']


for cat in cats:
    if not os.path.exists('./ShapeNetPointCloud' + cat):
        continue
    datadir = os.path.join('./ShapeNetPointCloud', cat, 'classify')
    outdir = os.path.join('./ShapeNetPointCloud', cat, 'classify_mat')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for f in sorted(os.listdir(datadir)):
        filepath = os.path.join(datadir,f)
        data = PlyData.read(filepath)['vertex'].data
        data = data.view(np.dtype('float32')).reshape(-1,7)
        data = data[:,0:3]
        t,mod,suffix=f.split('_')
        outfile = os.path.join(outdir,mod+'_'+t+'.mat')
        print(outfile)
