import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

mpl.style.use('bmh')

mpl.rcParams.update({'font.size': 10})
mpl.rc('xtick', labelsize=13)
mpl.rc('ytick', labelsize=13)


datasets = ['Wine', 'UJIndoorLoc', 'Sarcos', 'Synthetic']

bntl = [1617.57, 5998.94, 7241.86, 478.27]
ldbntl = [2472.35, 8266.20, 8768.89, 618.88]


n_groups = len(datasets)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.4
opacity = 0.8
capsize = 3


plt.bar(index + float(bar_width)/2, bntl, bar_width,
                alpha = opacity,
                color = 'C1',
                label = 'BNTL')



plt.bar(index + float(bar_width)/2 + bar_width, ldbntl, bar_width,
                alpha = opacity,
                color = 'C2',
                label = 'LDBNTL')



plt.xlabel('Datasets')
plt.ylabel('Time (sec)')
plt.xticks(index+bar_width, datasets, rotation=0)
plt.legend()
plt.savefig('timeplot.png')

plt.tight_layout()
plt.show()
