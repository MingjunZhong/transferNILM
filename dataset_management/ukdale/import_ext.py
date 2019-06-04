import numpy as np
import pandas as pd
import time
import os

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

#appliance_name = 'kettle'
#building = '1'
absolute_path = '/media/michele/Dati/ukdale/mingjun/'
buildings = [1, 2]
appliances = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
aggregate_mean = 522
aggregate_std = 814
save_path = '/media/michele/Dati/myUKDALE/'
save_path = '/home/michele/Desktop/'
print("\nNormalization parameters: ")
print("Mean and standard deviation values USED for AGGREGATE are:")
print("    Mean = {:d}, STD = {:d}".format(aggregate_mean, aggregate_std))



start_time = time.time()
for building in buildings:
    print('---------------------------Building: {} --------------------------'.format(building))
    for appliance_name in appliances:
        print('------------appliance: {} ------------'.format(appliance_name))
        path = absolute_path + appliance_name + '/'

        for idx, filename in enumerate(os.listdir(path)):
            if (appliance_name + '_building' + str(building) + '_train_mains') in filename and 'probnet' not in filename:
                name = filename
                aggregate = np.load(path + name).flatten()
                print("    loading files:")
                print('        ' + path + name)
            elif (appliance_name + '_building' + str(building) + '_train_target') in filename and 'probnet' not in filename:
                name = filename
                gt = np.load(path + name).flatten()
                print("    loading files:")
                print('        ' + path + name)

        assert aggregate.shape == gt.shape

        #aggregate[aggregate == -params_appliance[appliance_name]['mean']/params_appliance[appliance_name]['std']] = np.nan

        data = {
            'aggregate': aggregate,
            '{}'.format(appliance_name): gt,
        }

        df = pd.DataFrame(data)

        # de-Normalization and re-normalization
        df['aggregate'] = (df['aggregate']*params_appliance[appliance_name]['std']) + params_appliance[appliance_name]['mean']
        df['aggregate'] = (df['aggregate'] - aggregate_mean) / aggregate_std

        # Re-sampling
        ind = pd.date_range(0,  periods=df.shape[0], freq='6S')
        df.set_index(ind, inplace=True, drop=True)
        resample = df.resample('8S')
        df = resample.mean()

        # Save
        df.to_csv(save_path + appliance_name + '_test_' + 'uk-dale_' + 'H' + str(building) + '.csv', index=False)

        print("Size of test set is {:.3f} M rows (House {:d})."
              .format(df.shape[0] / 10 ** 6, int(building)))
        print('Mean and standard deviation values USED for ' + appliance_name + ' are:')
        print("    Mean = {:d}, STD = {:d}"
              .format(params_appliance[appliance_name]['mean'], params_appliance[appliance_name]['std']))

print("\nPlease find test set files in: " + save_path)
tot = int(int(time.time() - start_time) / 60)
print("\nTotal elapsed time: " + str(tot) + ' min')