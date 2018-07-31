import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

mpl.style.use('bmh')

mpl.rcParams.update({'font.size': 10})
mpl.rc('xtick', labelsize=13)
mpl.rc('ytick', labelsize=13)


datasets = ['Wine-Quality', 'UJIndoorLoc', 'Sarcos', 'Synthetic']

train_rmse = [0.64882, 0.15981, 0.02692, 0.05383]
train_std = [0.10496, 0.02315, 0.00879, 0.01654]

test_rmse = [0.65032, 0.21840, 0.02757, 0.08727]
test_std = [0.10701, 0.01424, 0.00889, 0.02901]

train_rmse_mh =[0.37225, 0.10111, 0.02549, 0.02107]
train_std_mh = [0.01866, 0.04320, 0.00477, 0.00593]

test_rmse_mh = [0.37908, 0.13446, 0.02636, 0.02266]
test_std_mh = [0.02225, 0.04481, 0.00482, 0.00719]

n_groups = len(datasets)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.20
opacity = 0.8
capsize = 3


plt.bar(index + float(bar_width)/2, train_rmse, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C1',
                yerr = train_std,
                label = 'Train Target Only')



plt.bar(index + float(bar_width)/2 + bar_width, train_rmse_mh, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C2',
                yerr = train_std_mh,
                label = 'Train TLMH')




plt.bar(index + float(bar_width)/2 + 2 * bar_width, test_rmse, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C3',
                yerr = test_std,
                label = 'Test Target Only')


plt.bar(index + float(bar_width)/2 + 3 * bar_width , test_rmse_mh, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C4',
                yerr = test_std_mh,
                label = 'Test TLMH')

plt.xlabel('Datasets')
plt.ylabel('RMSE')
plt.xticks(index+3*bar_width, datasets, rotation=0)
plt.legend()
plt.savefig('barplot.png')

plt.tight_layout()
plt.show()
# plt.savefig('barplt.png')
