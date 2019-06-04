import numpy as np
import pandas as pd
import time
import os
import re


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


start_time = time.time()
appliance_name = 'kettle'
print(appliance_name)

# UK-DALE path
path = '/media/michele/Dati/ukdale/'
save_path = '/home/michele/Desktop/'

aggregate_mean = 522
aggregate_std = 814

nrows = 10**5

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


print("Starting creating testset...")

for h in params_appliance[appliance_name]['houses']:

    print(path + 'house_' + str(h) + '/'
          + 'channel_' +
          str(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]) +
          '.dat')

    agg_df = load(path,
                  h,
                  appliance_name,
                  1,
                  nrows=nrows,
                  )

    df = load(path,
              h,
              appliance_name,
              params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)],
              nrows=nrows,
              )

    #for i in range(100):
    #    print(int(df['time'][i]) - int(agg_df['time'][i]))

    # Time conversion
    print(df.head())
    print(agg_df.head())
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    agg_df['time'] = pd.to_datetime(agg_df['time'], unit='ms')
    print(agg_df.head())
    print(df.head())

    df['aggregate'] = agg_df[appliance_name]
    cols = df.columns.tolist()
    del cols[0]
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    print(df.head())


    # Re-sampling
    ind = pd.date_range(0,  periods=df.shape[0], freq='6S')
    df.set_index(ind, inplace=True, drop=True)
    resample = df.resample('8S')
    df = resample.mean()

    print(df.head())

    # Normalization
    df['aggregate'] = (df['aggregate'] - aggregate_mean) / aggregate_std
    df[appliance_name] = \
        (df[appliance_name] - params_appliance[appliance_name]['mean']) / params_appliance[appliance_name]['std']

    # Save
    df.to_csv(save_path + appliance_name + '_test_' + 'uk-dale_' + 'H' + str(h) + '.csv', index=False)

    print("Size of test set is {:.3f} M rows (House {:d})."
          .format(df.shape[0] / 10 ** 6, h))

    del df


print("\nNormalization parameters: ")
print("Mean and standard deviation values USED for AGGREGATE are:")
print("    Mean = {:d}, STD = {:d}".format(aggregate_mean, aggregate_std))

print('Mean and standard deviation values USED for ' + appliance_name + ' are:')
print("    Mean = {:d}, STD = {:d}"
      .format(params_appliance[appliance_name]['mean'], params_appliance[appliance_name]['std']))

print("\nPlease find files in: " + save_path)
tot = int(int(time.time() - start_time) / 60)
print("\nTotal elapsed time: " + str(tot) + ' min')



