import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

mpl.style.use('bmh')

mpl.rcParams.update({'font.size': 10})
mpl.rc('xtick', labelsize=13)
mpl.rc('ytick', labelsize=13)


datasets = ['Wine-Quality', 'Wifi_1', 'Wifi_2', 'Wifi_3', 'Synthetic']

train_rmse = [0.64882,  0.11975, 0.15857, 0.12329, 0.06082]
train_std = [0.10496, 0.02414, 0.02525,  0.01732,  0.02253]

train_rmse_tl = [0.38000, 0.08620, 0.14002, 0.10837, 0.03229]
train_std_tl = [0.02372, 0.01866, 0.03052, 0.01316, 0.01391]

train_rmse_mh =[0.37225, 0.07596, 0.11516, 0.09571, 0.03218]
train_std_mh = [0.01866, 0.01149, 0.02520, 0.01386, 0.02012]

test_rmse = [0.65032, 0.14012, 0.19624, 0.14764, 0.06728]
test_std = [0.10701, 0.02082, 0.02075,  0.02094, 0.02662]

test_rmse_tl = [0.38194, 0.08058, 0.13594, 0.11176, 0.03254]
test_std_tl = [0.02315, 0.02225, 0.02283,  0.01837, 0.01364]

test_rmse_mh = [0.37908, 0.07904, 0.12152, 0.10617 , 0.03466]
test_std_mh = [0.02225, 0.01108, 0.02896,  0.01222, 0.01906]

n_groups = len(datasets)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.13
opacity = 0.8
capsize = 3


plt.bar(index + float(bar_width)/2, train_rmse, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C0',
                yerr = train_std,
                label = 'Train Target Only')

plt.bar(index + float(bar_width)/2 + bar_width, train_rmse_tl, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C1',
                yerr = train_std_tl,
                label = 'Train TL')

plt.bar(index + float(bar_width)/2 + 2*bar_width, train_rmse_mh, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C2',
                yerr = train_std_mh,
                label = 'Train TLMH')




plt.bar(index + float(bar_width)/2 + 3 * bar_width, test_rmse, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C3',
                yerr = test_std,
                label = 'Test Target Only')

plt.bar(index + float(bar_width)/2 + 4 * bar_width , test_rmse_tl, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C4',
                yerr = test_std_tl,
                label = 'Test TL')

plt.bar(index + float(bar_width)/2 + 5 * bar_width , test_rmse_mh, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C5',
                yerr = test_std_mh,
                label = 'Test TLMH')

plt.xlabel('Dataset')
plt.ylabel('RMSE')
plt.xticks(index+3*bar_width, datasets, rotation=0)
plt.title('UJIndoorLoc Dataset')
plt.legend()
plt.savefig('barplot.png')

plt.tight_layout()
plt.show()
# plt.savefig('barplt.png')
