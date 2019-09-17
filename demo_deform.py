import os
import cv2
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import sys
sys.path.append('Mesh_refinement')
from deformation.utils import *
from deformation.api import GCN
import trimesh

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 192, 'Number of units in hidden layer.')
flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.')

flags.DEFINE_string('data_root', 'demo', 'the path of img')
flags.DEFINE_string('category', 'chair', 'category (chair, table, plane)')
flags.DEFINE_string('suffix', '.png', 'the suffix of img file')
flags.DEFINE_string('basemesh_dirname', 'basemesh', 'the dirname of basemesh')
flags.DEFINE_string('finalmesh_dirname', 'finalmesh', 'the dirname of final output mesh')

def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        img[np.where(img[:,:,3]==0)] = 255
        img = cv2.resize(img, (224,224))
        img = img.astype('float32')/255.0
        return img[:,:,:3]
    else:
        img = cv2.resize(img, (224,224))
        print(img.shape)
        img = img.astype('float32')/255.0
        return img

#config the input and output directory
image_dir = FLAGS.data_root + '/' + FLAGS.category
basemesh_dir = FLAGS.data_root + '/' + FLAGS.category + '_' + FLAGS.basemesh_dirname
if not os.path.exists(basemesh_dir):
    print('Please generate base mesh firstly!!!')
    exit()
finalmesh_dir = FLAGS.data_root + '/' + FLAGS.category + '_' + FLAGS.finalmesh_dirname
if not os.path.exists(finalmesh_dir):
    os.mkdir(finalmesh_dir)

# Define placeholders(dict) and model
num_blocks = 3
num_supports = 2
placeholders = {
    'img_inp': tf.placeholder(tf.float32, shape=(224, 224, 3)),
    'labels': tf.placeholder(tf.float32, shape=(None, 6)), #(N,7)
    'features': tf.placeholder(tf.float32, shape=(None, 3)),
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'edges': tf.placeholder(tf.int32, shape=(None, 2)),
    'faces' : tf.placeholder(tf.int32, shape=(None, 3)),
    #'lape_idx': tf.placeholder(tf.float32, shape=(None,20)), 
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
model = GCN(placeholders, logging=True)

config=tf.ConfigProto()
#config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
dirname = 'Mesh_refinement/trained_models/%s'%FLAGS.category
model.load(sess, dirname)

for f in os.listdir(basemesh_dir):
    #load demo image
    basemesh_file = os.path.join(basemesh_dir, f)
    mod = f[:-4]
    image_path = os.path.join(image_dir, mod + FLAGS.suffix)
    img_inp = load_image(image_path)

    mesh = trimesh.load_mesh(basemesh_file)
    #vertices
    vertices = mesh.vertices
    vertices = vertices.astype('float32')
    #faces
    faces = mesh.faces
    faces = faces.astype('int32')
    #halfedges
    halfedges = mesh.edges
    edges = halfedges.astype('int32')

    #compute support
    coord = vertices
    vertex_size = len(coord)
    edge_size = len(edges)
    iden = sp.eye(vertex_size)
    adj = sp.coo_matrix( (np.ones(edge_size,dtype='float32'), (edges[:,0],edges[:,1])),
        shape=(vertex_size,vertex_size))
    support = [sparse_to_tuple(iden), normalize_adj(adj)]
    #update feed_dict
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: np.zeros([10,7])})
    feed_dict.update({placeholders['features']: coord})
    feed_dict.update({placeholders['img_inp']: img_inp})
    feed_dict.update({placeholders['edges']: edges})
    feed_dict.update({placeholders['faces']: faces})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['dropout']: 0.})
    feed_dict.update({placeholders['num_features_nonzero']: coord[1].shape})
    #run mesh deformation module
    out1,out2,out3 = sess.run([model.output1, model.output2, model.output3], feed_dict=feed_dict)
    #save final mesh
    finalmesh_file = finalmesh_dir + '/' + mod + '.obj'
    mesh = trimesh.Trimesh(out3, faces, vertex_normals=None, process=False)
    mesh.export(finalmesh_file)
    print('vertices:', out3.shape, 'faces:', faces.shape, finalmesh_file)

sess.close()
print('Mesh Deformation Finished!')