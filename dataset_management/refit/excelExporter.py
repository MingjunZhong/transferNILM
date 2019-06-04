from nilm.Arguments import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import pandas as pd

path = '/media/michele/Dati/CLEAN_REFIT_081116/'

# Change house number from here
building = 'House5'

filename = path + 'CLEAN_' + building + '.csv'

channels = [0, 1, 3, 4, 7, 8]


columns = [x+2 for x in channels]
columns_name = ['aggregate', 'fridge', 'washing_machine', 'dishwasher', 'microwave', 'kettle']

start = np.random.randint(0, 5*10**6)
print(str(start))
length = 10**5

csv = pd.read_csv(filename,
                  nrows=length,
                  skiprows=start,
                  names=columns_name,
                  usecols=columns,
                  dtype=int,
                  header=0,
                  parse_dates=True,
                  infer_datetime_format=True,
                  memory_map=True
                  )

# Plot csv
fig = plt.figure(num='Figure')
ax1 = fig.add_subplot(111)

ax1.plot(csv['aggregate'])
ax1.plot(csv['fridge'])
ax1.plot(csv['washing_machine'])
ax1.plot(csv['dishwasher'])
ax1.plot(csv['microwave'])
ax1.plot(csv['kettle'])


#ax1.set_title('{:}'.format(filename), fontsize=14, fontweight='bold')
ax1.set_ylabel('Power [W]')
ax1.set_xlabel('samples')
ax1.legend(columns_name)
ax1.grid()

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show(fig)


save_path = '/home/michele/Desktop/' + 'slice_from_' + building +'.csv'
csv.to_csv(save_path, index=False)
