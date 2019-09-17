import tensorflow as tf
from deformation.utils import *
from deformation.models import GCN
from deformation.fetcher import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_list', '', 'Data list test.')
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 192, 'Number of units in hidden layer.')
flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight for L2 loss.')
flags.DEFINE_string('checkpoint_dir', 'chair', 'load the preTrained model.')
flags.DEFINE_float('weight_edge', 0, 'Weight for edge loss')
flags.DEFINE_float('weight_normal', 0, 'Weight for normal loss.')
flags.DEFINE_bool('vertex_chamfer', True, 'If compute the cd (vertexes to surface).')

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

# Load data, initialize session
data = DataFetcher(FLAGS.data_list)
data.setDaemon(True) ####
data.start()
config=tf.ConfigProto()
#config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
dirname = FLAGS.checkpoint_dir
model.load(sess, dirname)

# Test graph model
test_loss = open(dirname+'/test_loss_record.txt', 'a')
test_loss.write('Start testing Chamfer distance (between mesh vertexes and surface points)')

test_number = data.number
all_loss = np.zeros(test_number,dtype='float32')
for iters in range(test_number):
    # Fetch testing data
    img_inp, label, pkl, mod_seq = data.fetch()
    feed_dict = construct_feed_dict(img_inp, pkl, label, placeholders)

    # Testing step
    cd, = sess.run([model.chamfer], feed_dict=feed_dict)
    all_loss[iters] = cd
    mean_loss = np.mean(all_loss[np.where(all_loss)])
    print('Iteration %d, Mean loss = %f, iter loss = %f, %d'%(iters+1, mean_loss, cd, data.queue.qsize()))
    test_loss.write('loss %f\n' % mean_loss)
    test_loss.flush()

    # Save for visualization
    out1, out2, out3, = sess.run([model.output1,model.output2,model.output3], feed_dict=feed_dict)
    export_img_mesh(img_inp, label, pkl, mod_seq, out1, out2, out3, dirname)
data.shutdown()
print('CNN-GCN Test Finished!')
