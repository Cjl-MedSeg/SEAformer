import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import *

def dice_loss(y_true, y_pred):
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred))
    loss = 1. - 2 * intersection / (union + K.epsilon())
    return loss

def jaccard_loss(y_true, y_pred):
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    loss = 1. - intersection / (union + K.epsilon())
    return loss

def BCE_IoU_loss(y_true, y_pred):
    ce_loss = binary_crossentropy(y_true, y_pred)
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    jaccard_loss = - tf.math.log((intersection + K.epsilon()) / (union + K.epsilon()))
    loss = ce_loss + jaccard_loss
    return loss

def edge_loss(y_true, y_pred):

    y_true_bd = MaxPooling2D((3, 3), strides=(1, 1), padding='same',data_format='channels_last')(1 - y_true)
    y_pred_bd = MaxPooling2D((3, 3), strides=(1, 1), padding='same',data_format='channels_last')(1 - y_pred)
    y_true_bd_ext = y_true_bd - (1 - y_true)
    y_pred_bd_ext = y_pred_bd - (1 - y_pred)
    y_true_pos = K.flatten(y_true_bd_ext)
    y_pred_pos = K.flatten(y_pred_bd_ext)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.75
    loss = 1. - (true_pos + K.epsilon())/(true_pos + alpha * false_neg + (1-alpha) * false_pos + K.epsilon())

    return loss



def iou_edge_loss(y_true, y_pred):

    alpha = 0.6
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    jaccard_loss = - tf.math.log((intersection + K.epsilon()) / (union + K.epsilon()))
    _edge_loss = edge_loss(y_true, y_pred)
    loss = (1 - alpha) * _edge_loss + alpha * jaccard_loss
    return loss


def multi_level_iou_edge_loss(y_true, y_pred):

    loss1 = iou_edge_loss(y_true, tf.expand_dims(y_pred[...,0], axis=-1))
    loss2 = iou_edge_loss(y_true, tf.expand_dims(y_pred[...,1], axis=-1))
    loss3 = iou_edge_loss(y_true, tf.expand_dims(y_pred[...,2], axis=-1))
    loss4 = iou_edge_loss(y_true, tf.expand_dims(y_pred[...,3], axis=-1))
    loss5 = iou_edge_loss(y_true, tf.expand_dims(y_pred[...,4], axis=-1))
    loss = (loss1 + loss2 + loss3 + loss4) / 4 + loss5

    return loss




