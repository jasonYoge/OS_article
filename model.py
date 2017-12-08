#!/usr/bin/python
# -*- coding: utf-8 -*-


from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import glob
from PIL import Image
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


W_SIZE = 105
H_SIZE = 105
D_SIZE = 3
batch_size = 10
TRAIN_DIR = os.path.join('../', 'DL_article', 'train')
VAL_DIR = os.path.join('../', 'DL_article', 'validation')
epoch = 3
file_size = 2000


def W_init(shape, name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def b_init(shape, name=None):
    """Initialize bias as in paper"""
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def get_sub_dir(dir):
    if os.path.exists(dir):
        paths = [x[0] for x in os.walk(dir)][1:]
    else:
        raise Exception('files path of dir not exists.')
    return paths


def get_file_names(dir):
    if os.path.exists(dir):
        file_glob = os.path.join(dir, '*.jpg')
        filenames = glob.glob(file_glob)
    else:
        raise Exception('file names of dir not exists.')
    return filenames


def create_model(w_size, h_size, d_size):
    input_shape = (w_size, h_size, d_size)
    left_input = Input(shape=input_shape, name='left_input')
    right_input = Input(shape=input_shape, name='right_input')

    # build convnet to use in each siamese 'leg'
    convnet = Sequential()
    convnet.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                       kernel_initializer=W_init, kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128, (7, 7), activation='relu',
                       kernel_initializer=W_init, bias_initializer=b_init, kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=W_init, kernel_regularizer=l2(2e-4),
                       bias_initializer=b_init))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=W_init, kernel_regularizer=l2(2e-4),
                       bias_initializer=b_init))
    convnet.add(Flatten())
    convnet.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3), kernel_initializer=W_init,
                      bias_initializer=b_init))

    # encode each of the two inputs into a vector with the convnet
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    # merge two encoded inputs with the l1 distance between them
    L1_distance = lambda x : K.abs(x[0] - x[1])
    both = merge([encoded_l, encoded_r], mode=L1_distance, output_shape=lambda x : x[0])
    prediction = Dense(1, activation='sigmoid', bias_initializer=b_init)(both)
    siamese_net = Model(input=[left_input, right_input], output=prediction)

    return siamese_net


def get_path_list(dir):
    list = []
    path = get_sub_dir(dir)
    for i in range(len(path)):
        list.append(get_file_names(path[i]))
    return list


def read_img(filename):
    if os.path.exists(filename):
        img_data = Image.open(filename)
        arr = np.asarray(img_data, dtype='float32')

    else:
        raise Exception('image path not exists for %s,' % filename)
    return arr


def load_data(file_list, batch_size=10):
    data_list = [[], []]
    label_list = []
    nb_classes = len(file_list)
    for i in range(batch_size):
        type_1 = rng.randint(nb_classes)
        type_2 = rng.randint(nb_classes)
        idx_1 = rng.randint(len(file_list[type_1]))
        idx_2 = rng.randint(len(file_list[type_2]))
        data_list[0].append(read_img(file_list[type_1][idx_1]))
        data_list[1].append(read_img(file_list[type_2][idx_2]))
        if type_1 == type_2:
            label_list.append(1)
        else:
            label_list.append(0)
    return data_list, label_list


if __name__ == '__main__':
    model = create_model(W_SIZE, H_SIZE, D_SIZE)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    train = get_path_list(TRAIN_DIR)
    val = get_path_list(VAL_DIR)
    for i in range(epoch):
        print 'epoch %d/%d' % (i, epoch)
        for j in range(file_size / batch_size):
            x, y = load_data(train, batch_size)
            loss = model.train_on_batch({'left_input': x[0],
                                         'right_input': x[1]}, y)
            print 'iteration %d, loss is %f' % (j, loss)
            if j % batch_size == 0:
                x_val, y_val = load_data(val, batch_size)
                print model.predict(x_val)
