import tensorflow as tf
from chamfer_distance import *

def laplace_coord(pred, placeholders):
    vertex = tf.concat([pred, tf.zeros([1,3])], 0)
    indices = tf.cast(placeholders['lape_idx'][:, :19], tf.int32)
    weights = placeholders['lape_idx'][:,-1]

    weights = tf.tile(tf.reshape(weights, [-1,1]), [1,3])
    laplace = tf.reduce_sum(tf.gather(vertex, indices), 1)
    laplace = tf.subtract(pred, tf.multiply(laplace, weights))
    return laplace

def laplace_loss(pred1, pred2, placeholders):
    # laplace term
    lap1 = laplace_coord(pred1, placeholders)
    lap2 = laplace_coord(pred2, placeholders)
    laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap1,lap2)), 1)) *1500 
    move_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(pred1, pred2)), 1)) *100
    return laplace_loss + move_loss

def unit(tensor):
    return tf.nn.l2_normalize(tensor, dim=1)

def cd_loss(pred,placeholders):
    gt_pt = placeholders['labels'][:, 0:3] # gt points
    gt_nm = placeholders['labels'][:, 3:6] # gt normals
    dist1,idx1,dist2,idx2 = nn_distance(gt_pt, pred)
    cd = (tf.reduce_mean(dist1) + tf.reduce_mean(dist2))
    return cd,dist1,idx1,dist2,idx2

def mesh_loss(pred, placeholders, weight_cd, weight_edge, weight_normal):
    gt_pt = placeholders['labels'][:, 0:3] # gt points
    gt_nm = placeholders['labels'][:, 3:6] # gt normals
    gt_classify = placeholders['labels'][:, -1] # gt_classify

    nod1 = tf.gather(pred, placeholders['edges'][:,0])
    nod2 = tf.gather(pred, placeholders['edges'][:,1])
    edge = tf.subtract(nod1, nod2)

    #edge length loss
    edge_length = tf.reduce_sum(tf.square(edge), 1)
    edge_loss = tf.reduce_mean(edge_length) * weight_edge

    # chamer distance
    dist1,idx1,dist2,idx2 = nn_distance(gt_pt, pred)
    point_loss = (tf.reduce_mean(dist1) + 0.55* tf.reduce_mean(dist2)) * weight_cd

    #normal cosine loss
    normal = tf.gather(gt_nm, tf.squeeze(idx2, 0))
    normal = tf.gather(normal, placeholders['edges'][:,0])
    cosine = tf.abs(tf.reduce_sum(tf.multiply(unit(normal), unit(edge)), 1))
    normal_loss = tf.reduce_mean(cosine) * weight_normal

    total_loss = point_loss + edge_loss + normal_loss
    return total_loss
