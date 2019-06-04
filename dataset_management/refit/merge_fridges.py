import pandas as pd
import matplotlib.pyplot as plt


def load(path, building, appliance, channel):
    # load csv
    file_name = path + 'CLEAN_House' + str(building) + '.csv'
    #print('inside function')
    #print(appliance)
    #print(channel)
    single_csv = pd.read_csv(file_name,
                             header=0,
                             names=['aggregate', appliance],
                             # index_col=0,
                             usecols=[2, channel+2],
                             dtype={'aggregate': int, "appliance": int},
                             # dtype=np.int32,
                             # engine='c',
                             na_filter=False,
                             parse_dates=True,
                             infer_datetime_format=True,
                             memory_map=True)

    return single_csv

appliance_name='fridge'

params_appliance = {
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': {
            '4': [1, 2, 3]

        },
    }
}

path = '/media/michele/Dati/CLEAN_REFIT_081116/'

single_csv = pd.read_csv(path + 'CLEAN_House4.csv',
                             header=0,
                             names=['aggregate', 'appliance1', 'appliance2', 'appliance3'],
                             # index_col=0,
                             usecols=[2, 3, 4, 5],
                             #dtype={int},
                             # dtype=np.int32,
                             # engine='c',
                             na_filter=False,
                             parse_dates=True,
                             infer_datetime_format=True,
                             memory_map=True)

single_csv[appliance_name] = single_csv.iloc[:, 1:].sum(1)
single_csv = single_csv.drop('appliance1', 1)
single_csv = single_csv.drop('appliance2', 1)
single_csv = single_csv.drop('appliance3', 1)

single_csv.plot()
plt.show()

"""
path = '/media/michele/Dati/CLEAN_REFIT_081116/'


single_csv = pd.read_csv(path + 'CLEAN_House6.csv',
                         header=0,
                         nrows=5 * 10 **6)

single_csv.to_csv('a.csv', index=False)
"""