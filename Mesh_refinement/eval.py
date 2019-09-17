import os
import sys
import time, datetime
import numpy as np
import tensorflow as tf
from deformation.utils import *
from deformation.api import GCN
from deformation.fetcher import *
sys.path.append('external')
from deformation.chamfer_distance import nn_distance
from tf_approxmatch import approx_match, match_cost
import trimesh

def compute_fscore(dist_forward, dist_backward, threshold=1e-4):
    recall = ((dist_forward<threshold).astype('f4').sum(1))/dist_forward.shape[1]
    precision = ((dist_backward<threshold).astype('f4').sum(1))/dist_backward.shape[1]
    fscore = 200*precision*recall/(precision+recall+1e-6)
    fscore = np.mean(fscore)
    return fscore

class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_list', '', 'Data list (testing).')
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 192, 'Number of units in hidden layer.')
flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight for L2 loss.')
flags.DEFINE_string('checkpoint_dir', 'chair', 'load the preTrained model.')
flags.DEFINE_float('weight_edge', 0, 'Weight for edge loss')
flags.DEFINE_float('weight_normal', 0, 'Weight for normal loss.')
flags.DEFINE_bool('vertex_chamfer', False, 'If compute the cd (vertexes to surface).')
flags.DEFINE_integer('npoints_cd', 10000, 'Number of sampled points to compute CD.')
flags.DEFINE_integer('npoints_emd', 10000, 'Number of sampled points to compute EMD.')
flags.DEFINE_float('fscore_th', 0.0001, 'Threshold for computing fscore.')

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
    #'lape_idx': tf.placeholder(tf.float32, shape=(None,20)), #for laplace term
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
model = GCN(placeholders, logging=True)

# Load data
data = DataFetcher(FLAGS.data_list)
data.setDaemon(True) ####
data.start()
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
dirname = FLAGS.checkpoint_dir
model.load(sess, dirname)

# Initialize session
# xyz1:dataset_points * 3, xyz2:query_points * 3
xyz1=tf.placeholder(tf.float32,shape=(None, 3))
xyz2=tf.placeholder(tf.float32,shape=(None, 3))
# chamfer distance
dist1,idx1,dist2,idx2 = nn_distance(xyz1, xyz2)
# earth mover distance, notice that emd_dist return the sum of all distance
match = approx_match(xyz1, xyz2)
emd_dist = match_cost(xyz1, xyz2, match)

# Test graph model
CD = AverageValueMeter()
EarthMD = AverageValueMeter()
FSCORE = AverageValueMeter()
CD.reset()
EarthMD .reset()
FSCORE.reset()

evaluation_log = open(dirname+'/evaluation_record.txt', 'a')
evaluation_log.write('Start evaluating CD, EMD, F-score: %s \n'%FLAGS.checkpoint_dir)
test_number = data.number
for iters in range(test_number):
    t0 = time.time()
    # Fetch testing data
    img_inp, label, pkl, mod_seq = data.fetch()
    feed_dict = construct_feed_dict(img_inp, pkl, label, placeholders)

    # Testing step
    vertices, = sess.run([model.output3], feed_dict=feed_dict)
    faces = pkl[2]
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=None, process=False)
    points_pred = trimesh.sample.sample_surface(mesh, FLAGS.npoints_cd)[0]
    points_pred = np.require(points_pred,'float32', 'C')
    print(points_pred.shape)
    points_gt = label[:FLAGS.npoints_cd, 0:3]
    points_gt = np.require(points_gt, 'float32', 'C')

    d1,i1,d2,i2, = sess.run([dist1,idx1,dist2,idx2], feed_dict={xyz1:points_gt, xyz2:points_pred})
    cd = np.mean(d1) + np.mean(d2)
    match_dist = sess.run([emd_dist], feed_dict={xyz1:points_gt[:FLAGS.npoints_emd, :], xyz2:points_pred[:FLAGS.npoints_emd, :]})
    earth_md = match_dist[0]
    fscore = compute_fscore(d1, d2, threshold=FLAGS.fscore_th)
    CD.update(cd)
    EarthMD.update(earth_md)
    FSCORE.update(fscore)
    print('[%d/%d] mean cd:  %f, mean  emd: %f, fscore: %f, time: %f'
        %(iters, test_number, CD.avg*1000, EarthMD.avg*0.01, FSCORE.avg, time.time()-t0))
evaluation_log.write('chamfer distance (x 1000) : %f \n'% (CD.avg*1000))
evaluation_log.write('earth movers distance(x 0.01): %f \n'% (EarthMD.avg*0.01))
evaluation_log.write('fscore (x 100): %f with threshold: %f \n'%(FSCORE.avg*100, FLAGS.fscore_th))
evaluation_log.close()
sess.close()
data.shutdown()
print('Mesh Deformation Evaluation Finished!')