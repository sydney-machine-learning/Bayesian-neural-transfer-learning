import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

mpl.style.use('bmh')

mpl.rcParams.update({'font.size': 10})
mpl.rc('xtick', labelsize=13)
mpl.rc('ytick', labelsize=13)


datasets = ['UJIndoorLoc', 'Sarcos', 'Synthetic']

train_rmse = [0.15981, 0.02692, 0.05383]
train_std = [0.02315, 0.00879, 0.01654]

test_rmse = [0.21840, 0.02757, 0.08727]
test_std = [0.01424, 0.00889, 0.02901]

train_rmse_bntl =[0.10111, 0.02549, 0.02107]
train_std_bntl = [0.04320, 0.00477, 0.00593]

test_rmse_bntl = [0.13446, 0.02636, 0.02266]
test_std_bntl = [0.04481, 0.00482, 0.00719]

train_rmse_ld =[0.16554, 0.02242, 0.04024]
train_std_ld = [0.01838, 0.00374, 0.01252]

test_rmse_ld = [0.24502, 0.02239, 0.04064]
test_std_ld = [0.03239, 0.00398, 0.01222]

train_rmse_ldbntl =[0.10801, 0.01768, 0.01294]
train_std_ldbntl = [0.02229, 0.00250, 0.00524]

test_rmse_ldbntl = [0.14384, 0.01739, 0.01301]
test_std_ldbntl = [0.02174, 0.00247, 0.00553]


n_groups = len(datasets)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.1
opacity = 0.8
capsize = 3


plt.bar(index + float(bar_width)/2, train_rmse, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C1',
                yerr = train_std,
                label = 'Train Target Only')



plt.bar(index + float(bar_width)/2 + bar_width, train_rmse_bntl, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C2',
                yerr = train_std_bntl,
                label = 'Train BNTL')

plt.bar(index + float(bar_width)/2 + 2 * bar_width, train_rmse_ld, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C3',
                yerr = train_std_ld,
                label = 'Train LD Target Only')


plt.bar(index + float(bar_width)/2 + 3 * bar_width , train_rmse_ldbntl, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C4',
                yerr = train_std_ldbntl,
                label = 'Train LDBNTL')




plt.bar(index + float(bar_width)/2 + 4 * bar_width, test_rmse, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C5',
                yerr = test_std,
                label = 'Test Target Only')


plt.bar(index + float(bar_width)/2 + 5 * bar_width , test_rmse_bntl, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C6',
                yerr = test_std_bntl,
                label = 'Test BNTL')

plt.bar(index + float(bar_width)/2 + 6 * bar_width, test_rmse_ld, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C7',
                yerr = test_std_ld,
                label = 'Test LD Target Only')


plt.bar(index + float(bar_width)/2 + 7 * bar_width , test_rmse_ldbntl, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C9', capsize=capsize),
                color = 'C8',
                yerr = test_std_ldbntl,
                label = 'Test LDBNTL')

plt.xlabel('Datasets')
plt.ylabel('RMSE')
plt.xticks(index+2*bar_width, datasets, rotation=0)
plt.legend()
plt.savefig('barplot.png')

plt.tight_layout()
plt.show()
# plt.savefig('barplt.png')
