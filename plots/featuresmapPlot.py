from Arguments import *
from cnnModel import get_model, weights_loader
from dataset_management.refit.dataset_infos import *
from keras.models import Model
from keras.layers import Input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re


appliance_name = args.appliance_name

# Looking for the selected test set
for filename in os.listdir(args.datadir + appliance_name):
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
loadname_test = args.datadir + appliance_name + '/' + test_filename
log('Loading from: ' + loadname_test)

df = pd.read_csv(loadname_test,
                 nrows=10**6,
                 #skiprows=3450*10**3,
                 #header=0
                 )

house = int(re.search(r'\d+', test_filename).group())
path = '/media/michele/Dati/CLEAN_REFIT_081116/' + 'CLEAN_House' + str(house) + '.csv'
#print(house)
columns = houses[str(house)]
columns.remove(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(house)])
columns_names.remove(appliance_name)

c_names = []
for i in columns:
    string = 'Appliance' + str(i)
    if string == 'Appliance10':
        string = 'Issues'
    c_names.append(string)

df_alias = pd.read_csv(path,
                       nrows=10**6,
                       header=0,
                       #index_col=False,
                       usecols=c_names,
                       #names=columns_names,
                       dtype=int,
                       #skiprows=3450*10**3,
                       )[c_names]
df_alias.columns = [columns_names]


#start = 109210  # kettle
#start = 159270  # dishwasher
#start = 151850  start = 959750 # washing machine
#start = 188600  # fridge
#start = np.random.randint(0, 10**6-599)
#start = 691000 # fridge footprint
#start = 226500 # microwave

start = 188400

length = params_appliance[appliance_name]['windowlength']

inp = np.array(df.iloc[start:start+length, 0]).reshape(1, length)
tar = np.array(df.iloc[start:start+length, 1]).reshape(1, length)

appliances = {}
for name in columns_names:
    if name == appliance_name:
        pass
    else:
        appliances['{}'.format(name)] = df_alias.loc[start:start+length, name].values

log(inp.shape)
log(inp.ndim)

"""
# Customize input
inp = np.zeros((1, length), dtype=int)
inp = inp + 0.45
noise = np.random.normal(0.1, 0.11, (1, length))

for i in range(0, 500, 80):
    inp[0, i:i+50] = i*0.009

inp = inp + noise
tar = np.zeros((1, length), dtype=int)
"""

# ------------------------------ KERAS NETWORK - from cnnModel.py ------------------------------------------------------

uno = Input(shape=(1, length),
            #batch_shape=None
            )

model = get_model(uno,
                  params_appliance[args.appliance_name]['windowlength'],
                  n_dense=args.dense_layers
                  )[0]

y = model.outputs

# Load path depending on the model kind
if args.transfer:
    param_file = '../models/cnn_s2p_' + appliance_name + '_transf_' + args.cnn + '_pointnet_model'
else:
    param_file = '../models/cnn_s2p_' + args.appliance_name + '_pointnet_model'
# Loading weigths
weights_loader(model, param_file)
# ----------------------------------------------------------------------------------------------------------------------


# ---------------------------------------- Plot CNN weights ------------------------------------------------------------
cnn = 0
fig1, ax = plt.subplots(nrows=5, ncols=1)
fig1.subplots_adjust(hspace=0.5)

for idx, layer in enumerate(model.layers):
    #print(layer.name)
    if 'conv2d_'in layer.name or 'cnn' in layer.name:
        extracted_weights = layer.get_weights()
        #model_untrained.layers[idx].set_weights(extracted_weights)
        #model_untrained.layers[idx].trainable = False
        w = np.array(extracted_weights[0][:, :, 0, :])

        log(w.shape)
        weights_arr = np.array(w).reshape(w.shape[-1], w.shape[0])

        colormap0 = ax[cnn].imshow(weights_arr,
                                   cmap='magma',
                                   aspect='auto')

        fig1.colorbar(colormap0, ax=ax[cnn])

        ax[cnn].set_title('CNN_{}'.format(cnn+1))
        cnn += 1

if args.transfer:
    fig1.suptitle('{} - CNN layers weigths'.format(args.cnn), fontsize=16, fontweight='bold')
else:
    fig1.suptitle('{} - CNN layers weigths'.format(args.appliance_name), fontsize=16, fontweight='bold')

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show(fig1)


# -------------------------------------- Plot neurons weights ----------------------------------------------------------
layer_name = 'output'

weights = model.get_layer(layer_name).get_weights()[0]
#print(weights.shape)
biases = model.get_layer(layer_name).get_weights()[1]
#print(biases.shape)

fig2, ax = plt.subplots(nrows=1, ncols=1)

cmap = 'magma'
matrix = np.transpose(weights)
colormap0 = ax.imshow(matrix,
                        cmap=cmap,
                        aspect=25,
                        #vmin=0,
                        vmax=0.5
                        )

fig2.colorbar(colormap0, ax=ax, orientation='vertical', aspect=25, shrink=0.5)

fig2.suptitle('{}'.format(args.appliance_name), fontsize=16, fontweight='bold')
ax.set_title('Final dense layers weights')
ax.set_xlabel('weights')
ax.set_yticks([])

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show(fig2)

# ----------------------------------------------------- Prediction------------------------------------------------------
output = model.predict(inp.reshape(1, -1, length),
                       batch_size=None,
                       steps=None
                       )


# CNN Prediction
layer_name = 'conv2d_5'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

intermediate_output = intermediate_layer_model.predict(inp.reshape(1, -1, length),
                                                       batch_size=None,
                                                       steps=None
                                                       )


# --------------------------------------- PLOT features ----------------------------------------------------------------

# Normilzation
inp = inp * 814 + 522
tar = tar * params_appliance[appliance_name]['std'] + params_appliance[appliance_name]['mean']
out = output[0][0] * params_appliance[appliance_name]['std'] + params_appliance[appliance_name]['mean']
if out < 0:
    out = 0
log("Output point {:.6f}".format(out))



fig3, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

ax[0].plot(inp.reshape(-1))
ax[0].plot(tar.reshape(-1))
ax[0].plot(299, out, c='r', marker='x', markersize=9)
legend = ['Input windows', 'Ground truth', 'Prediction']

for name in columns_names:
    if name == appliance_name:
        pass
    else:
        ax[0].plot(appliances[name], linewidth=0.8)
        legend.append(name)

ax[0].grid()
ax[0].set_title('Input Window')
ax[0].set_xlabel('samples')
ax[0].set_ylabel('W')
ax[0].legend(legend)

some_matrix = np.transpose(intermediate_output[0, 0, :, :])
cmap = 'magma'
#cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)

colormap = ax[1].imshow(some_matrix,
                        cmap=cmap,
                        aspect='auto',
                        #vmin=-0.005,
                        #vmax=0.20
                        )

ax[1].grid()
ax[1].set_title('Output Features')
ax[1].set_xlabel('samples')
ax[1].set_ylabel('Filters')
fig3.colorbar(colormap, ax=ax[1], orientation='horizontal')

if args.transfer:
    fig3.suptitle('Features from the last (learnt from {:}) CNN layer - (dense {:})'
              .format(args.cnn, appliance_name), fontsize=16, fontweight='bold')
else:
    fig3.suptitle('Features from the last (learnt from {:}) CNN layer - (dense {:})'
              .format(appliance_name, appliance_name), fontsize=16, fontweight='bold')

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()


# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])


"""
# -------------------------------------------- PLOT XXX ----------------------------------------------------------------

start = 0
length = 1

df1 = pd.read_csv(loadname_test,
                 nrows=length,
                 skiprows=start,
                 #header=0
                 )

inp1 = np.array(df1.iloc[start:start+length, 0]).reshape(1, length)
tar1 = np.array(df1.iloc[start:start+length, 1]).reshape(1, length)

fig4, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

ax[0].plot(inp1.reshape(-1))
ax[0].plot(tar1.reshape(-1))

ax[0].grid()
ax[0].set_title('Input Window')
ax[0].set_xlabel('samples')
ax[0].set_ylabel('W')
#ax[0].legend()

some_matrix = np.transpose(intermediate_output[0, 0, :, :])
cmap = 'magma'
colormap = ax[1].imshow(some_matrix,
                        cmap=cmap,
                        aspect='auto',
                        #vmin=-0.005,
                        #vmax=0.20
                        )

ax[1].grid()
ax[1].set_title('Output Features')
ax[1].set_xlabel('samples')
ax[1].set_ylabel('Filters')
fig4.colorbar(colormap, ax=ax[1], orientation='horizontal')
#fig4.suptitle('Feaures from the last ({:}) CNN layer - (appliance {:})'
#              .format(args.cnn, appliance_name), fontsize=16, fontweight='bold')



some_matrix1 = np.transpose(intermediate_output[0, 0, :, :])
cmap = 'magma'
colormap = ax[2].imshow(some_matrix,
                        cmap=cmap,
                        aspect='auto',
                        #vmin=-0.005,
                        #vmax=0.20
                        )

#ax[1].grid()
#ax[1].set_title('Output Features')
#ax[1].set_xlabel('samples')
#ax[1].set_ylabel('Filters')
#fig4.colorbar(colormap, ax=ax[1], orientation='horizontal')
#fig4.suptitle('Feaures from the last ({:}) CNN layer - (appliance {:})'
#              .format(args.cnn, appliance_name), fontsize=16, fontweight='bold')


mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()
"""

