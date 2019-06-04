import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
plt.rcParams.update({'font.size': 13})
import pandas as pd


def load(path, building, appliance, channel, nrows=None):
    # load csv
    file_name = path + 'house_' + str(building) + '/' + 'channel_' + str(channel) + '.dat'
    single_csv = pd.read_csv(file_name,
                             sep=' ',
                             #header=0,
                             names=['time', appliance],
                             dtype={'time': str, "appliance": int},
                             #parse_dates=['time'],
                             #date_parser=pd.to_datetime,
                             nrows=nrows,
                             usecols=[0, 1],
                             engine='python'
                             )
    return single_csv


appliance_name = 'washingmachine'
path = '/media/michele/Dati/ukdale/'

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


# Change house number from here
building = 1


file_name = path +\
            'house_' + str(building) + '/' +\
            'channel_' +\
            str(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses']
                .index(building)]) +\
            '.dat'

# Number of sample to plot
chunksize = 10**3

agg_df = load(path,
              building,
              appliance_name,
              1,
              nrows=chunksize,
              )

df = load(path,
          building,
          appliance_name,
          params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(building)],
          nrows=chunksize,
          )


"""
df = pd.read_csv(file_name,
            nrows=chunksize,
            index_col=False,
            names=['aggregate', appliance_name],
            usecols=[1, 2],
            header=0
            )

df = pd.read_csv(file_name,
            nrows=chunksize,
            index_col=False,
            names=['aggregate', appliance_name],
            usecols=[1, 2],
            header=0
            )
"""

df['aggregate'] = agg_df[appliance_name]
del agg_df

fig = plt.figure(num='Figure {:}'.format(appliance_name))
ax1 = fig.add_subplot(111)

ax1.plot(df['time'], df['aggregate'])
ax1.plot(df['time'], df[appliance_name])
#ax1.plot(chunk['Unix'], chunk['Appliance2'])
#ax1.plot(chunk['Unix'], chunk['Appliance3'])
#ax1.plot(chunk['Unix'], chunk['Appliance4'])
#ax1.plot(chunk['Unix'], chunk['Appliance5'])
#ax1.plot(chunk['Unix'], chunk['Appliance6'])
#ax1.plot(chunk['Unix'], chunk['Appliance7'])
#ax1.plot(chunk['Unix'], chunk['Appliance8'])
#ax1.plot(chunk['Unix'], chunk['Appliance9'])
#ax1.plot(chunk['Unix'], chunk['Issues']*1000)

#ax1.plot(chunk['Aggregate'])  # light blue
#ax1.plot(chunk['Appliance1'])  # orange
#ax1.plot(chunk['Appliance2'])  # green
#ax1.plot(chunk['Appliance3'])  # red
#ax1.plot(chunk['Appliance4'])  # violette
#ax1.plot(chunk['Appliance8'])  # brown
#ax1.plot(chunk['Appliance6'])  # pink
#ax1.plot(chunk['Appliance7'])  # gray
#x1.plot(chunk['Appliance8'])  # darkyellow
#ax1.plot(chunk['Appliance9'])  # azure

#ax1.set_title('{:}'.format(file_name), fontsize=14, fontweight='bold',
                  # y=1.08
#                  )

ax1.set_ylabel('Power [W]')
ax1.set_xlabel('samples')
ax1.legend(['aggregate', 'appliance'])
#ax1.grid()

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show(fig)

