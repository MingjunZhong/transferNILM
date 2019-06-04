from Arguments import *
from dataset_management.refit.dataset_infos import *
from keras.models import Model
from keras.layers import Input
from cnnModel import get_model, weights_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
import os
import re


# Looking for the selected test set
for filename in os.listdir(args.datadir + args.appliance_name):
        if args.test_type == 'train' and 'TRAIN' in filename.upper():
            test_filename = filename
        elif args.test_type == 'uk' and 'UK' in filename.upper():
            test_filename = filename
        elif args.test_type == 'test' and 'TEST' in\
                filename.upper() and 'TRAIN' not in filename.upper() and 'UK' not in filename.upper():
            test_filename = filename
        elif args.test_type == 'val' and 'VALIDATION' in filename.upper():
            test_filename = filename


log('File for test: ' + test_filename)
loadname_test = args.datadir + args.appliance_name + '/' + test_filename
log('Loading from: ' + loadname_test)

# Number of sample to plot
#start = 109210  # kettle
#start = 159270  # dishwasher
#start = 151850  start = 959750 # washing machine
#start = 188600  2208200 # fridge
#start = np.random.randint(0, 10**6-599)
#start = 691000 # fridge footprint
#start = 226500 # microwave

start = 2208200

length = params_appliance[args.appliance_name]['windowlength']
df = pd.read_csv(loadname_test,
                 nrows=length,
                 skiprows=start,
                 #header=0
                 )

inp = np.array(df.ix[:, 0]).reshape(1, length)
tar = np.array(df.ix[:, 1]).reshape(1, length)

# --------------------------------- KERAS NETWORK - from file ----------------------------------------------------------
uno = Input(shape=(1, length),
            # batch_shape=None
            )
model = get_model(uno,
                  params_appliance[args.appliance_name]['windowlength'],
                  n_dense=args.dense_layers
                  )[0]
y = model.outputs

apps = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']

cnn = 1
fig, ax = plt.subplots(nrows=6, ncols=1, sharex=True)
fig.subplots_adjust(hspace=0.5)

inp_norm = inp * 814 + 522
tar_norm = tar * params_appliance[args.appliance_name]['std'] + params_appliance[args.appliance_name]['mean']
ax[0].plot(inp_norm.reshape(-1))
ax[0].plot(tar_norm.reshape(-1))
ax[0].grid()
#ax[0].set_title('Input Window')
ax[0].set_xlabel('samples')
ax[0].set_ylabel('Power [W]')
ax[0].legend(['Input window', 'Ground truth'])

for app in apps:

    # Load path depending on the model kind
    param_file = '../models/cnn_s2p_' + app + '_pointnet_model'
    weights_loader(model, param_file)

    layer_name = 'conv2d_5'
    #layer_name = 'cnn5'

# --------------------------------------------- Predictions ------------------------------------------------------------
# CNN Prediction
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(inp.reshape(1, -1, length),
                                                           batch_size=None,
                                                           steps=None
                                                           )

# --------------------------------------- PLOT features ----------------------------------------------------------------

    some_matrix = np.transpose(intermediate_output[0, 0, :, :])
    cmap = 'magma'
    colormap = ax[cnn].imshow(some_matrix,
                            cmap=cmap,
                            aspect='auto',
                            # vmin=-0.005,
                            # vmax=0.20
                            )
    ax[cnn].set_title('CNN_{}'.format(app))
    ax[cnn].grid()
    ax[cnn].set_ylabel('Channels')
    cnn += 1


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(colormap, cax=cbar_ax)
fig.suptitle('Latent features comparison on aggregate containing {:} footprint'
             .format(args.appliance_name), fontsize=16, fontweight='bold')

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()

