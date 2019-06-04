import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


def window_stack(a, stepsize=1, width=3):
    return np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width) )


def remove_space(string):
    return string.replace(" ","")


params_appliance = {
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
        'houses': [1, 2, 3],
        'channels': [6, 10, 9],
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [1, 2, 3],
        'channels': [20, 7, 13],
    }
}


start_time = time.time()
app_list = ['microwave', 'fridge', 'dishwasher', 'washingmachine']


# REDD path
path = '/media/michele/Dati/REDD/'
save_path = '/home/michele/Desktop/'

sample_seconds = 8 # fixed
#start = args.start_time   # data starting date
#end = args.end_time     # data end date
debug = False
nrows=None

for app in app_list:
    appliance_name = app
    print('Appliance name: ' + appliance_name)
    for h in params_appliance[appliance_name]['houses']:
        print('\n')
        print(path + 'house_' + str(h) + '/'
              + 'channel_' +
              str(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]) +
              '.dat')


        # read data
        mains1_df = pd.read_table(path + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                 str(1) + '.dat',
                                  sep="\s+",
                                  nrows=nrows,
                                  usecols=[0, 1],
                                  names=['time', 'mains1'],
                                  dtype={'time': str},
                                  )

        mains2_df = pd.read_table(path + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                 str(2) + '.dat',
                                  sep="\s+",
                                  nrows=nrows,
                                  usecols=[0, 1],
                                  names=['time', 'mains2'],
                                  dtype={'time': str},
                                  )
        app_df = pd.read_table(path + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                 str(params_appliance[appliance_name]['channels']
                                     [params_appliance[appliance_name]['houses'].index(h)]) + '.dat',
                               sep="\s+",
                               nrows=nrows,
                               usecols=[0, 1],
                               names=['time', app],
                               dtype={'time': str},
                               )

        """
        d1 = np.random.random_integers(0,7000,[4000,1])
        d2 = np.random.random_integers(0,7000,[2000,1])

        dfA = pd.DataFrame(d1)
        dfB = pd.DataFrame(d2)

        dfA.columns = ['data1']
        dfB.columns = ['data2']

        dfA['time'] = pd.date_range('1970-01-01 00:01:00', periods=dfA.shape[0], freq='1S')
        dfB['time'] = pd.date_range('1970-01-01 00:00:00', periods=dfB.shape[0], freq='1S')

        dfA.set_index('time', inplace=True)
        dfB.set_index('time', inplace=True)

        dfA1 = dfA.between_time('00:00:00', '00:09:00')
        dfA2 = dfA.between_time('00:14:00', '00:16:00')

        dfB1 = dfB.between_time('00:00:00', '00:12:00')
        dfB2 = dfB.between_time('00:15:00', '00:16:00')

        df1 = pd.concat([dfA1, dfA2])
        df2 = pd.concat([dfB1, dfB2])

        #df1.plot()
        #df2.plot()

        #plt.plot(df1.values)
        #plt.plot(df2.values)
        #plt.show()

        df_align = df1.join(df2, how='outer').resample('2S').mean().fillna(method='backfill', limit=1)
        df_align = df_align.dropna()
        df_align.plot()



        plt.plot(df_align['data1'].values)
        plt.plot(df_align['data2'].values)
        plt.show()
        """

        # Aggregate
        #mains1_df = mains1_df.set_index(mains1_df.columns[0])
        #mains1_df.index = pd.to_datetime(mains1_df.index, unit='s')
        mains1_df['time'] = pd.to_datetime(mains1_df['time'], unit='s')
        #print(mains1_df.count())
        #mains1_df['OVER 5 MINS'] = (mains1_df['time'].diff()).dt.seconds > 1
        #plt.plot(mains1_df['OVER 5 MINS'])
        #print("mains1_df:")
        #print(mains1_df.head())
        #mains2_df = mains2_df.set_index(mains2_df.columns[0])
        #mains2_df.index = pd.to_datetime(mains2_df.index, unit='s')
        mains2_df['time'] = pd.to_datetime(mains2_df['time'], unit='s')
        #print(mains2_df.count())
        #print("mains2_df:")
        #print(mains2_df.head())
        # merging two mains
        #plt.plot(mains1_df['time'],mains1_df['mains1'])
        #plt.show()
        mains1_df.set_index('time', inplace=True)
        mains2_df.set_index('time', inplace=True)
        #mains_df = mains1_df.join(mains2_df, how='outer').interpolate(method='time')
        mains_df = mains1_df.join(mains2_df, how='outer')
        #print("mains_df:")
        #print(mains_df.head())
        mains_df['aggregate'] = mains_df.iloc[:].sum(axis=1)

        resample = mains_df.resample(str(sample_seconds)+'S').mean()
        #mains_df = resample.mean()

        mains_df.reset_index(inplace=True)
        #print(mains_df.count())
        #mains_df['OVER 5 MINS'] = (mains_df['time'].diff()).dt.seconds > 1
        #plt.plot(mains_df['OVER 5 MINS'])
        #mains_df.columns = ['mains1', 'mains2']
        #mains_df.plot()
        #plt.show()
        #print("mains_df:")
        #print(mains_df.head())

        # resampling 8 sec
        #resample = mains_df.resample(str(sample_seconds)+'S')
        #mains_df = resample.mean()

        #ind = pd.date_range(0, periods=df.shape[0], freq='6S')
        #df.set_index(ind, inplace=True, drop=True)


        # deleting original separate mains
        del mains_df['mains1'], mains_df['mains2']

        if debug:
            print("mains_df:")
            print(mains_df.head())
            plt.plot(mains_df['time'], mains_df['aggregate'])
            plt.show()

        # Appliance
        #app_df = app_df.set_index(app_df.columns[0])
        #app_df.index = pd.to_datetime(app_df.index, unit='s')
        app_df['time'] = pd.to_datetime(app_df['time'], unit='s')
        #app_df.columns = [appliance_name]
        if debug:
            print("app_df:")
            print(app_df.head())
            plt.plot(app_df['time'], app_df[appliance_name])
            plt.show()

        # the timestamps of mains and appliance are not the same, we need to align them
        # 1. join the aggragte and appliance dataframes;
        # 2. interpolate the missing values;
        mains_df.set_index('time', inplace=True)
        app_df.set_index('time', inplace=True)

        df_align = mains_df.join(app_df, how='outer').\
                                resample(str(sample_seconds)+'S').mean().fillna(method='backfill', limit=1)
        df_align = df_align.dropna()


        df_align.reset_index(inplace=True)
        print(df_align.count())
        #df_align['OVER 5 MINS'] = (df_align['time'].diff()).dt.seconds > 9
        #df_align.plot()
        #plt.plot(df_align['OVER 5 MINS'])
        #plt.show()

        del mains1_df, mains2_df, mains_df, app_df, df_align['time']

        mains = df_align['aggregate'].values
        app_data = df_align[appliance_name].values
        #plt.plot(np.arange(0, len(mains)), mains, app_data)
        #plt.show()

        if debug:
            # plot the dtaset
            print("df_align:")
            print(df_align.head())
            plt.plot(df_align['aggregate'].values)
            plt.plot(df_align[appliance_name].values)
            plt.show()

        # Normilization ----------------------------------------------------------------------------------------------
        aggregate_mean = 522
        aggregate_std = 814
        mean = params_appliance[appliance_name]['mean']
        std = params_appliance[appliance_name]['std']

        df_align['aggregate'] = (df_align['aggregate'] - aggregate_mean) / aggregate_std
        df_align[appliance_name] = (df_align[appliance_name] - mean) / std

        # Save to csv
        df_align.to_csv(save_path + appliance_name + '_test_' + 'redd_' + 'H' + str(h) + '.csv', index=False)
        print("Size of test set is {:.3f} M rows (House {:d})."
              .format(df_align.shape[0] / 10 ** 6, h))

        del df_align


print("\nNormalization parameters: ")
print("Mean and standard deviation values USED for AGGREGATE are:")
print("    Mean = {:d}, STD = {:d}".format(aggregate_mean, aggregate_std))

print('Mean and standard deviation values USED for ' + appliance_name + ' are:')
print("    Mean = {:d}, STD = {:d}"
      .format(params_appliance[appliance_name]['mean'], params_appliance[appliance_name]['std']))

print("\nPlease find files in: " + save_path)
tot = int(int(time.time() - start_time) / 60)
print("\nTotal elapsed time: " + str(tot) + ' min')

