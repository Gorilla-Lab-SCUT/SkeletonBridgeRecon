import os
import sys
import numpy as np
import scipy.sparse as sp
import trimesh
import cv2

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_adj(features):
    # normalizes symetric, binary adj matrix such that sum of each row is 1 
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def construct_feed_dict(img_inp, pkl, labels, placeholders):
    """Construct feed dictionary."""
    coord = pkl[0]
    edges = pkl[1]
    faces = pkl[2]
    ##lape_idx = pkl[3]

    vertex_size = len(coord)
    edge_size = len(edges)
    iden = sp.eye(vertex_size)
    adj = sp.coo_matrix((np.ones(edge_size,dtype='float32'),
                        (edges[:,0],edges[:,1])),shape=(vertex_size,vertex_size))
    support = [sparse_to_tuple(iden), normalize_adj(adj)]

    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: coord})
    feed_dict.update({placeholders['img_inp']: img_inp})
    feed_dict.update({placeholders['edges']: edges})
    feed_dict.update({placeholders['faces']: faces})
    ##feed_dict.update({placeholders['lape_idx']: lape_idx})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: coord[1].shape})
    feed_dict.update({placeholders['dropout']: 0.})
    return feed_dict

def export_img_mesh(img_inp, label, pkl, mod_seq, out1, out2, out3, savedir):
    if not os.path.exists(savedir+'/img'):
        os.mkdir(savedir+'/img')
    if not os.path.exists(savedir+'/predict'):
        os.mkdir(savedir+'/predict')
    img_file = savedir+'/img/%s.png'%mod_seq 
    cv2.imwrite(img_file, img_inp*255)

    mesh_file = savedir+'/predict/%s.obj'%mod_seq
    vertices = out3
    faces = pkl[2]
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=None, process=False)
    mesh.export(mesh_file)
    print('vertices:', out3.shape, 'faces:', faces.shape, mesh_file)