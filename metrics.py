from keras import backend as K
from keras.metrics import binary_accuracy

def acc(y_true, y_pred):
    return binary_accuracy(y_true, y_pred)

def sens(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_true = K.round(y_true)
    true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    total_pos = K.sum(K.abs(y_true), axis=[1, 2, 3])
    return true_pos / K.clip(total_pos, K.epsilon(), None)

recall = sens

def spec(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_true = K.round(y_true)
    true_neg = K.sum(K.abs((1. - y_true) * (1. - y_pred)), axis=[1, 2, 3])
    total_neg = K.sum(K.abs(1. - y_true), axis=[1, 2, 3])
    return true_neg / K.clip(total_neg, K.epsilon(), None)

def dice(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_true = K.round(y_true)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
    dice = 2 * intersection / K.clip(union, K.epsilon(), None)
    return dice

def iou(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_true = K.round(y_true)
    intersection = K.sum(K.abs(y_true * K.round(y_pred)), axis=[1, 2, 3])
    union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
    iou = intersection / K.clip(union - intersection, K.epsilon(), None)
    return iou