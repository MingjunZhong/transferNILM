from Arguments import *
from Logger import log
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, Reshape
from keras.utils import print_summary, plot_model
import numpy as np
import keras.backend as K
import os
import tensorflow as tf
import h5py
import argparse


def get_model(appliance, input_tensor, window_length, transfer_dense=False, transfer_cnn=False,
              cnn='kettle', n_dense=1, pretrainedmodel_dir='./models/'):

    reshape = Reshape((-1, window_length, 1),
                      )(input_tensor)

    cnn1 = Conv2D(filters=30,
                  kernel_size=(10, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  )(reshape)

    cnn2 = Conv2D(filters=30,
                  kernel_size=(8, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  )(cnn1)

    cnn3 = Conv2D(filters=40,
                  kernel_size=(6, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  )(cnn2)

    cnn4 = Conv2D(filters=50,
                  kernel_size=(5, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  )(cnn3)

    cnn5 = Conv2D(filters=50,
                  kernel_size=(5, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  )(cnn4)

    flat = Flatten(name='flatten')(cnn5)

    d = Dense(1024, activation='relu', name='dense')(flat)

    if n_dense == 1:
        label = d
    elif n_dense == 2:
        d1 = Dense(1024, activation='relu', name='dense1')(d)
        label = d1
    elif n_dense == 3:
        d1 = Dense(1024, activation='relu', name='dense1')(d)
        d2 = Dense(1024, activation='relu', name='dense2')(d1)
        label = d2

    d_out = Dense(1, activation='linear', name='output')(label)

    model = Model(inputs=input_tensor, outputs=d_out)

    session = K.get_session()

    if transfer_dense:
        log("Transfer learning...")
        log("...loading an entire pre-trained model")
        weights_loader(model, pretrainedmodel_dir+'/cnn_s2p_' + appliance + '_pointnet_model')
        model_def = model
    elif transfer_cnn and not transfer_dense:
        log("Transfer learning...")
        log('...loading a ' + appliance + ' pre-trained-cnn')
        cnn_weights_loader(model, cnn, pretrainedmodel_dir)
        model_def = model
        for idx, layer1 in enumerate(model_def.layers):
            if hasattr(layer1, 'kernel_initializer') and 'conv2d' not in layer1.name and 'cnn' not in layer1.name:
                log('Re-initialize: {}'.format(layer1.name))
                layer1.kernel.initializer.run(session=session)

    elif not transfer_dense and not transfer_cnn:
        log("Standard training...")
        log("...creating a new model.")
        model_def = model
    else:
        raise argparse.ArgumentTypeError('Model selection error.')

    # Printing, logging and plotting the model
    print_summary(model_def)
    # plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True, rankdir='TB')

    # Adding network structure to both the log file and output terminal
    files = [x for x in os.listdir('./') if x.endswith(".log")]
    with open(max(files, key=os.path.getctime), 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model_def.summary(print_fn=lambda x: fh.write(x + '\n'))

    # Check weights slice
    for v in tf.trainable_variables():
        if v.name == 'conv2d_1/kernel:0':
            cnn1_weights = session.run(v)
    return model_def, cnn1_weights


def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))


def cnn_weights_loader(model_to_fill, cnn_appliance, pretrainedmodel_dir):
    log('Loading cnn weights from ' + cnn_appliance)    
    weights_path = pretrainedmodel_dir+'/cnn_s2p_' + cnn_appliance + '_pointnet_model' + '_weights.h5'
    if not os.path.exists(weights_path):
        print('The directory does not exist or you do not have the files for trained model')
        
    f = h5py.File(weights_path, 'r')
    log(f.visititems(print_attrs))
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    for name in layer_names:
        if 'conv2d_' in name or 'cnn' in name:
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]

            model_to_fill.layers[int(name[-1])+1].set_weights(weight_values)
            log('Loaded cnn layer: {}'.format(name))

    f.close()
    print('Model loaded.')


def weights_loader(model, path):
    log('Loading cnn weights from ' + path)
    model.load_weights(path + '_weights.h5')





