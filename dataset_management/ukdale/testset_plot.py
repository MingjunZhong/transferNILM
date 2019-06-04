import matplotlib.pyplot as plt
import pandas as pd

params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128,
        'houses': [1, 2],
        'channels': [10, 8],
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [1, 2],
        'channels': [13, 15],
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [1, 2],
        'channels': [12, 14],
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [1, 2],
        'channels': [6, 13],
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [1, 2],
        'channels': [5, 12],
    }
}

appliance_name = 'kettle'
building = 2


chunksize = None
path = path = '/media/michele/Dati/myUKDALE/'

test_filename = path + appliance_name + '_test_' + 'uk-dale_' + 'H' + str(building) + '.csv'

chunk = pd.read_csv(test_filename,
                    nrows=chunksize,
                    names=['aggregate', appliance_name],
                    header=0
                    )

# de-normalization
chunk['aggregate'] = chunk['aggregate'] * 822 + 522
chunk[appliance_name] = chunk[appliance_name] * params_appliance[appliance_name]['std'] \
                  + params_appliance[appliance_name]['mean']


# Figure
fig = plt.figure(num='Figure {:}'.format(appliance_name))
ax1 = fig.add_subplot(111)

ax1.plot(chunk['aggregate'])
ax1.plot(chunk[appliance_name])

ax1.grid()
ax1.set_title('{:}'.format(test_filename), fontsize=14, fontweight='bold')
ax1.set_ylabel('Power [W]')
ax1.set_xlabel('samples')
ax1.legend(['aggregate', appliance_name])

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show(fig)


