from tensorflow.keras.layers import *
from tensorflow.keras import Input, Model
import tensorflow as tf
from tensorflow.keras.applications import densenet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy
from SEAformer.utils import PatchEmbed, dense_mlpblock, SEA_block, dense_convblock, transition_block

def stage(vit, cnn, pre_train, cnndown_blocks, vitdown_blocks, use_focus, grown_rate, embed_dim = None, sea = True, name='stage1'):  # 112,64   # 112,32

    if grown_rate != 32:

        cnn_first_channel = 16
        cnn_rate = 16
        mlp_rate = 16 if grown_rate == 16 else 32

    else:
        cnn_first_channel = 32
        cnn_rate = 32
        mlp_rate = 32

    vit_steam = PatchEmbed(vit, embed_dim = embed_dim ,use_focus = use_focus)

    if embed_dim is not None:
        cnn = Conv2D(cnn_first_channel, 3, padding='same', kernel_initializer='he_normal')(cnn)

    if pre_train is not None and grown_rate != 32:
        conv_steam_skip = pre_train(cnn)
    else:
        conv_steam_skip = dense_convblock(cnn, cnndown_blocks, cnn_rate)


    conv_steam = transition_block(conv_steam_skip, 0.5)

    _, H, W, C = conv_steam.shape

    vit_steam_skip = Lambda(SEA_block)([vit_steam, conv_steam, sea])

    vit_steam = dense_mlpblock(vit_steam_skip, vitdown_blocks, mlp_rate)
    vit_steam = tf.reshape(vit_steam, [-1, H, W, C*2])

    conv_steam = conv_steam  + vit_steam_skip

    return vit_steam, conv_steam, conv_steam_skip, vit_steam_skip



def last_stage(vit, cnn, pre_train, cnndown_blocks, vitdown_blocks, grown_rate, sea = True):

    if grown_rate != 32:
        cnn_rate = 16
        mlp_rate = 16 if grown_rate == 16 else 32

    else:
        cnn_rate = 32
        mlp_rate = 32

    vit_steam = dense_mlpblock(vit, vitdown_blocks, mlp_rate)

    if pre_train is not None and grown_rate != 32:
        conv_steam = pre_train(cnn)
    else:
        conv_steam = dense_convblock(cnn, cnndown_blocks, cnn_rate)

    _, H, W, C = conv_steam.shape

    vit_steam = tf.reshape(vit_steam,[-1, H, W, C])
    vit_steam_skip = Lambda(SEA_block)([vit_steam, conv_steam, sea])

    conv_steam = vit_steam_skip + conv_steam
    conv_steam = Dropout(0.4)(conv_steam)

    return conv_steam



def UpBlock(x, x1, filter, cnnup_block, grown_rate):

    cnn_rate = 16 if grown_rate != 32 else 32

    x = Conv2D(filter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(x))
    x = concatenate([x1, x], axis=-1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter // 2, 3, padding='same', kernel_initializer='he_normal')(x)
    x = dense_convblock(x, cnnup_block, cnn_rate)

    return x


def head(x):

    x = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, 1, name='out5')(x)
    x = Activation('sigmoid')(x)

    return x


def Multi_level_supervised_learning(out_list):

    out1 = UpSampling2D(size=(2,2))(Conv2D(1, 1, name='out1')(out_list[0]))
    out2 = UpSampling2D(size=(4,4))(Conv2D(1, 1, name='out2')(out_list[1]))
    out3 = UpSampling2D(size=(8,8))(Conv2D(1, 1, name='out3')(out_list[2]))
    out4 = UpSampling2D(size=(16,16))(Conv2D(1, 1, name='out4')(out_list[3]))
    out5 = out_list[4]
    out = concatenate([Activation('sigmoid')(out1), Activation('sigmoid')(out2), Activation('sigmoid')(out3), Activation('sigmoid')(out4), out5], axis=-1,name = 'out')

    return out


def SEAFormer(pretrained_weights=None, input_shape=(224, 224, 3), use_focus = True, grown_rate = None):

    print("model:")
    for key, val in locals().items():
        if not val == None and not key == "kwargs":
            print("\t", key, "=", val)
    print("-------------------------------")

    if grown_rate != 32:
        if grown_rate ==16:

            print('Use conv rate is 16, mlp rate is 16')
            cnndown_blocks = [3, 6, 12, 24, 48]
            vitdown_blocks = [2, 4, 8, 16, 32]
            cnnup_block = [16, 8, 4, 2]

        else:
            print('Use conv rate is 16, mlp rate is 32')
            cnndown_blocks = [3, 6, 12, 24, 48]
            vitdown_blocks = [1, 2, 4, 8, 16]
            cnnup_block = [16, 8, 4, 2]

        cnn_steam = densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        Cnn3_premodel = Model(inputs=cnn_steam.get_layer('conv2_block1_0_bn').input, outputs=cnn_steam.get_layer('conv2_block6_concat').output, name="cnn3")  # 56,64->56,256
        Cnn4_premodel = Model(inputs=cnn_steam.get_layer('conv3_block1_0_bn').input, outputs=cnn_steam.get_layer('conv3_block12_concat').output, name="cnn4")  # 28,128->28,512
        Cnn5_premodel = Model(inputs=cnn_steam.get_layer('conv4_block1_0_bn').input, outputs=cnn_steam.get_layer('conv4_block24_concat').output,name="cnn5")  # 14,256->14,1024


    else :
        print('Use conv rate is 32, mlp rate is 32')
        cnndown_blocks = [1, 3, 6, 12, 24]
        vitdown_blocks = [1, 2, 4, 8, 16]      #32
        cnnup_block = [8, 4, 2, 1]

        '''
        Not using pretrained weights
        '''
        Cnn3_premodel = None
        Cnn4_premodel = None
        Cnn5_premodel = None

    inputs = Input(shape=input_shape)


    vit1, cnn1, skip1, out1 = stage(inputs, inputs, None, cnndown_blocks[0], vitdown_blocks[0], use_focus, grown_rate, embed_dim= 32)  # 112,32
    vit2, cnn2, skip2, out2 = stage(vit1, cnn1, None, cnndown_blocks[1], vitdown_blocks[1], use_focus, grown_rate)  # 56,64
    vit3, cnn3, skip3, out3 = stage(vit2, cnn2, Cnn3_premodel, cnndown_blocks[2], vitdown_blocks[2],use_focus, grown_rate)  # 28,128
    vit4, cnn4, skip4, out4 = stage(vit3, cnn3, Cnn4_premodel, cnndown_blocks[3], vitdown_blocks[3],use_focus, grown_rate)  # 14,256
    cnn5 = last_stage(vit4, cnn4, Cnn5_premodel, cnndown_blocks[4], vitdown_blocks[4], grown_rate)  # 14,1024


    up6 = UpBlock(cnn5, skip4, 512, cnnup_block[0], grown_rate)  # 28,512
    up7 = UpBlock(up6, skip3, 256, cnnup_block[1], grown_rate)  # 56,256
    up8 = UpBlock(up7, skip2, 128, cnnup_block[2], grown_rate)  # 112,128
    up9 = UpBlock(up8, skip1, 64, cnnup_block[3], grown_rate)  # 224,64
    out = head(up9)
    final_out = Multi_level_supervised_learning([out1, out2, out3, out4, out])

    model = Model(inputs, final_out, name='SEAFormer')
    model.summary()

    print('SEAFormer is loading..........\n'*3)

    model.compile(optimizer=Adam(learning_rate = 0.0001),
                  loss=binary_crossentropy,
                  metrics=binary_accuracy)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


SEAFormer(pretrained_weights=None, input_shape=(224, 224, 3), use_focus = True, grown_rate = 16)