import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import pandas as pd

params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128,
        'houses': [],
        'channels': [10, 8],
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [1, 2, 3],
        'channels': [11, 6, 16],
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [1, 2, 3],
        'channels': [5, 9, 7, 18],
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [1, 2, 3, 5],
        'channels': [6, 10, 9],
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [1, 2, 3, 5],
        'channels': [19, 7, 13, 8],
    }
}

# Change house number from here

appliance_name = 'washingmachine'
building = 3
print(appliance_name)
print(str(building))

# REDD path
path = '/media/michele/Dati/REDD/'
save_path = '/home/michele/Desktop/'

file_name = path +\
            'house_' + str(building) + '/' +\
            'channel_' +\
            str(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses']
                .index(building)]) +\
            '.dat'


# Number of sample to plot
mains_df = pd.read_table(path + '/' + 'house_' + str(building) + '/' + 'channel_13' + '.dat',
                         sep="\s+")
mains_df = mains_df.set_index(mains_df.columns[0])
mains_df.index = pd.to_datetime(mains_df.index, unit='s')
mains_df.columns = ['uno']

app_df = pd.read_table(path + '/' + 'house_' + str(building) + '/' + 'channel_14' + '.dat',
                       sep="\s+")
app_df = app_df.set_index(app_df.columns[0])
app_df.index = pd.to_datetime(app_df.index, unit='s')
app_df.columns = ['due']


plt.plot(mains_df['uno'].values)
plt.plot(app_df['due'].values)


#df_align = pd.DataFrame(mains_df['uno'])
#df_align['due'] = app_df['due']
#print(df_align.head())
#df_align.plot()
"""
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
"""
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()

