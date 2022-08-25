from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import backend

def group_conv(x, filters, kernel, stride, groups):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(x)[channel_axis]
    nb_ig = in_channels // groups
    nb_og = filters // groups

    gc_list = []
    for i in range(groups):
        if channel_axis == -1:
            x_group = Lambda(lambda z: z[:, :, :, i * nb_ig: (i + 1) * nb_ig])(x)
        else:
            x_group = Lambda(lambda z: z[:, i * nb_ig: (i + 1) * nb_ig, :, :])(x)
        gc_list.append(Conv2D(filters=nb_og, kernel_size=kernel, strides=stride,padding='same', use_bias=False)(x_group))  # 对每组特征图进行单独卷积

    return Concatenate(axis=channel_axis)(gc_list)


def PatchEmbed(x, embed_dim = None, use_focus = True):

    _, _, _, C = x.shape

    if use_focus:
        if embed_dim is not None:
            x = concatenate([x[:, ::2, ::2, :], x[:, 1::2, ::2, :], x[:, ::2, 1::2, :], x[:, 1::2, 1::2, :]], -1)
            x = group_conv(x, embed_dim, 3, 1, 4)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        else:
            x = concatenate([x[:, ::2, ::2, :], x[:, 1::2, ::2, :], x[:, ::2, 1::2, :], x[:, 1::2, 1::2, :]], -1)
            x = group_conv(x, C, 1, 1, 4)
            x = Activation('relu')(x)

    elif use_focus == False:
        if embed_dim is not None:
            x = Conv2D(filters=embed_dim, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        else:
            x = AveragePooling2D(pool_size=(2, 2))(x)
    return x


def mlp_block(x, growth_rate, drop_rate=0.1):
    x1 = BatchNormalization()(x)
    x1 = Activation('relu')(x1)
    x1 = Dense(4 * growth_rate, kernel_initializer='he_normal')(x1)
    x1 = Dense(growth_rate, kernel_initializer='he_normal')(x1)
    x1 = Dropout(rate = drop_rate)(x1)
    x = Concatenate()([x, x1])
    return x

def dense_mlpblock(x, blocks, grown_rate):
    _, H, W, C = x.shape
    x = tf.reshape(x,[-1, H * W, C])
    for i in range(blocks):
        x = mlp_block(x, grown_rate)
    return x


def conv_block(x, growth_rate):
    x1 = BatchNormalization()(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = Concatenate()([x, x1])
    return x

def dense_convblock(x, blocks, grown_rate):
    for i in range(blocks):
        x = conv_block(x, grown_rate)
    return x


def transition_block(x, reduction):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1, kernel_initializer='he_normal')(x)
    return x


def select_aggregate(feat_map):
    A = tf.reduce_sum(feat_map, axis=-1, keepdims=True)
    a = tf.reduce_mean(A, axis=[1, 2], keepdims=True)
    M = tf.cast(A > a, tf.float32)
    return M

def SEA_block(inputs):  # vit cnn

    x, x1, SEA = inputs[0], inputs[1], inputs[2]

    _, _, _, C = x.shape
    x = AveragePooling2D(pool_size=3, strides=1, padding='same')(x) - x
    y_sig1 = Activation('sigmoid')(x)

    y_sig = Activation('sigmoid')(x1)
    y = MaxPooling2D(pool_size=3, strides=1, padding='same', data_format='channels_last')(1 - y_sig)
    y = y - (1 - y_sig)
    m = select_aggregate(y_sig)

    y_reduce = tf.reduce_mean(y, axis=-1, keepdims=True)
    y1 = MaxPooling2D(pool_size=3, strides=1, padding='same', data_format='channels_last')(1 - y_reduce)
    y1 = (y1 - (1 - y_reduce))
    y_threshold = tf.reduce_sum(y1, axis=[1, 2], keepdims=True)
    y_threshold = tf.squeeze(y_threshold, axis=-1)

    ym = y * m
    
    threshold_list = []
    for i in range(y.shape[-1]):
        y_slice = ym[..., i]
        y_slice = tf.reduce_sum(y_slice, axis=[1, 2], keepdims=True)
        d = tf.cast(y_slice > y_threshold, tf.float32)
        threshold_list.append(d)

    threshold = Concatenate()(threshold_list)
    threshold = tf.expand_dims(threshold, axis=1)
    y_weigh = ym * threshold

    y_weigh = Activation('sigmoid')(tf.reduce_mean(tf.concat([y_weigh, y_sig1], axis=-1), axis=-1, keepdims=True))

    if SEA:
        out =  (x + y) * y_weigh
    else:
        out = (x + y)

    return out
