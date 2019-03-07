import numpy as np
from keras.models import Model,Sequential
from keras.utils.data_utils import get_file
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.optimizers import Adam,RMSprop,SGD
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras.initializers import he_normal
from keras.layers import Dropout
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Deconvolution2D, Cropping2D
from keras.layers import Input, Add, Dropout, Permute, add,Activation
from keras.initializers import he_normal


def encoder_block(conv_num, encoder_num,filters_num):
    encoding_conv_layers = []
    for i in range(conv_num):
        encoding_conv_layers.append(Conv2D(filters_num,kernel_size = (3, 3),
                                           padding = "same",activation = "relu",kernel_initializer = he_normal(seed=None), bias_initializer='zeros'
                                           ,name ="encoder."+str(encoder_num)+".layer."+str(i+1)))
    encoding_conv_layers.append(MaxPooling2D(name = "encoder."+str(encoder_num)))
    return encoding_conv_layers

def build_full_fcn8(image_dim, weights=None):   
    
    FCN8 = Sequential()
    FCN8.add(Permute((1,2,3),input_shape = (image_dim,image_dim,3)))
    #Building Encoder Blocks 
    for l in encoder_block(conv_num=2, encoder_num= 1,filters_num =64) :
        FCN8.add(l)
    for l in encoder_block(conv_num=2, encoder_num= 2,filters_num =128) :
        FCN8.add(l)
    for l in encoder_block(conv_num=3,encoder_num= 3,filters_num =256) :
        FCN8.add(l)
    for l in encoder_block(conv_num=3,encoder_num= 4,filters_num =512) :
        FCN8.add(l)
    for l in encoder_block(conv_num=3,encoder_num= 5,filters_num =512) :
        FCN8.add(l)
    fc6 = Conv2D(4096,kernel_size=(7,7),padding = "same",activation = "relu",name = "fc6")
    fc7 = Conv2D(4096,kernel_size=(1,1),padding = "same",activation = "relu",name = "fc7")
    score_fr = Convolution2D(2,kernel_size=(1,1),padding="same",activation="relu",name = "score_fr")
    upscore2 = Deconvolution2D(2,kernel_size=(4,4),strides = (2,2),padding = "valid",activation=None,name = "upscore2")
    FCN8.add(fc6)
    FCN8.add(Dropout(0.5))
    FCN8.add(fc7)
    FCN8.add(Dropout(0.5))
    FCN8.add(score_fr)
    FCN8.add(upscore2)
    FCN8.add(Cropping2D(cropping=((0,2),(0,2))))
    #skip Connection1 
    score_pool4 = Convolution2D(2,kernel_size=(1,1),padding = "same",activation=None, name = "score_pool4")
    fuse_pool4 = add(inputs = [score_pool4(FCN8.layers[14].output),FCN8.layers[-1].output])
    upscore_pool4 = Deconvolution2D(2,kernel_size=(4,4),strides = (2,2),padding = "valid",activation=None,name = "upscore_pool4")(fuse_pool4)
    crop4 = Cropping2D(cropping=((0,2),(0,2)))(upscore_pool4)
    #skip Connection2
    score_pool3 = Convolution2D(2,kernel_size=(1,1),padding = "same",activation=None, name = "score_pool3")
    fuse_pool3 = add(inputs = [score_pool3(FCN8.layers[10].output),crop4])
    upscore8 = Deconvolution2D(2,kernel_size=(16,16),strides = (8,8),padding = "valid",activation=None,name = "upscore8")(fuse_pool3)
    score = Cropping2D(cropping = ((0,8),(0,8)))(upscore8)
   
    softmax_score = Activation('softmax')(score)

    fcn8 = Model(FCN8.input, softmax_score)
    return fcn8 