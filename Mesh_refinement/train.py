import tensorflow as tf
from deformation.utils import *
from deformation.models_weight import GCN
from deformation.fetcher import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_list', '', 'Data list (training).')
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 192, 'Number of units in hidden layer.')
flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight for L2 loss.')
flags.DEFINE_bool('load_model', False, 'If load preTrained model.')
flags.DEFINE_string('checkpoint_dir', 'chair_e300_n2', 'save model and load the preTrained model.')
flags.DEFINE_float('weight_edge', 300, 'Weight for edge loss')
flags.DEFINE_float('weight_normal', 1, 'Weight for normal loss.')
flags.DEFINE_bool('vertex_chamfer', False, 'If compute the cd (vertexes to surface).')

# Define placeholders(dict) and model
num_blocks = 3
num_supports = 2
placeholders = {
    'img_inp': tf.placeholder(tf.float32, shape=(224, 224, 3)),
    'labels': tf.placeholder(tf.float32, shape=(None, 6)), #(N,7)
    'features': tf.placeholder(tf.float32, shape=(None, 3)),
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'edges': tf.placeholder(tf.int32, shape=(None, 2)),
    'faces': tf.placeholder(tf.int32, shape=(None, 3)),
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
if FLAGS.load_model:
    model.load(sess, dirname)

# Train graph model
train_loss = open(dirname + '/train_loss_record.txt', 'a')
train_loss.write('Start training, lr =  %f\n'%(FLAGS.learning_rate))
train_number = data.number
for epoch in range(FLAGS.epochs):
    all_loss = np.zeros(train_number,dtype='float32') 
    for iters in range(train_number):
        # Fetch training data
        img_inp, label, pkl, mod_seq = data.fetch()
        feed_dict = construct_feed_dict(img_inp, pkl, label, placeholders)

        # Training step
        _, dists = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)
        all_loss[iters] = dists
        mean_loss = np.mean(all_loss[np.where(all_loss)])
        print mod_seq
        print 'Epoch %d, Iteration %d'%(epoch + 1,iters + 1)
        print 'Mean loss = %f, iter loss = %f, %d'%(mean_loss,dists,data.queue.qsize())
        #if (iters+1)%10000==0:
        #    model.save(sess, dirname)
    # Save model
    model.save(sess, dirname)
    train_loss.write('Epoch %d, loss %f\n'%(epoch+1, mean_loss))
    train_loss.flush()

data.shutdown()
print 'CNN-GCN Optimization Finished!'
