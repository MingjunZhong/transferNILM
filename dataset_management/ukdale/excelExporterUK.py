from nilm.Arguments import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import pandas as pd

path = '/media/michele/Dati/myUKDALE/'

# Change house number from here
building = '2'



apps = ['fridge', 'washingmachine', 'dishwasher', 'microwave', 'kettle']

start = np.random.randint(0, 1*10**6)
start=0
print(str(start))
length = None


filename = path + 'kettle/' + 'kettle' + '_test_' + 'uk-dale_' + 'H' + str(building) + '.csv'
main = pd.read_csv(filename,
                  nrows=length,
                  skiprows=start,
                  names=['aggregate'],
                  usecols=[0],
                  header=0,
                  parse_dates=True,
                  infer_datetime_format=True,
                  memory_map=True
                  )

main['aggregate'] = (main['aggregate'] * 814) + 522


for app in apps:
    filename = path + app + '/' + app + '_test_' + 'uk-dale_' + 'H' + str(building) + '.csv'
    csv = pd.read_csv(filename,
                      nrows=length,
                      skiprows=start,
                      names=[app],
                      usecols=[1],
                      header=0,
                      parse_dates=True,
                      infer_datetime_format=True,
                      memory_map=True
                      )

    csv[app]=csv[app]*params_appliance[app]['std']+params_appliance[app]['mean']
    main[app] = csv[app]

    del csv



# Plot csv
fig = plt.figure(num='Figure')
ax1 = fig.add_subplot(111)

ax1.plot(main['aggregate'])
ax1.plot(main['fridge'])
ax1.plot(main['washingmachine'])
ax1.plot(main['dishwasher'])
ax1.plot(main['microwave'])
ax1.plot(main['kettle'])


#ax1.set_title('{:}'.format(filename), fontsize=14, fontweight='bold')
ax1.set_ylabel('Power [W]')
ax1.set_xlabel('samples')
ax1.legend(['aggregate', 'fridge', 'washing_machine', 'dishwasher', 'microwave', 'kettle'])
ax1.grid()

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show(fig)


save_path = '/home/michele/Desktop/' + 'slice_from_' + building +'.csv'
csv.to_csv(save_path, index=False)
