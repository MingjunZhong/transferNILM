from cnnModel import get_model, weights_loader
from Arguments import *
from dataset_management.refit.dataset_infos import *
from keras.layers import Input
import numpy as np
import matplotlib.pyplot as plt

length = params_appliance[args.appliance_name]['windowlength']


# ------------------------------ KERAS NETWORK - from cnnModel.py ------------------------------------------------------

uno = Input(shape=(1, length))
model = get_model(uno,
                  params_appliance[args.appliance_name]['windowlength'],
                  transfer_cnn=args.transfer,
                  cnn=args.cnn,
                  )

y = model.outputs

# Load path depending on the model kind
if args.transfer:
    param_file = '../models/cnn_s2p_' + args.appliance_name + '_transf_' + args.cnn + '_pointnet_model'
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
print(weights.shape)
biases = model.get_layer(layer_name).get_weights()[1]
print(biases.shape)

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

