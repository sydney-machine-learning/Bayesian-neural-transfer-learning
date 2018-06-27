import numpy as np
import matplotlib.pyplot as plt

source_index = [index + 1 for index in range(10)]

train_rmse = np.zeros((len(source_index)))
test_rmse = np.zeros((len(source_index)))

train_rmse_trf = np.zeros((len(source_index)))
test_rmse_trf = np.zeros((len(source_index)))

train_std_trf = np.zeros((len(source_index)))
test_std_trf = np.zeros((len(source_index)))

train_std = np.zeros((len(source_index)))
test_std = np.zeros((len(source_index)))


for index in source_index:

    rmsetrain_trf = np.genfromtxt('synthetic_data '+str(index)+'/targettrftrainrmse.csv')
    rmsetest_trf = np.genfromtxt('synthetic_data '+str(index)+'/targettrftestrmse.csv')

    burnin = int(0.1 * rmsetrain_trf.shape[0])
    stdtrain_trf = rmsetrain_trf[burnin: ].std(axis=0)
    stdtest_trf = rmsetest_trf[burnin: ].std(axis=0)
    rmsetrain_trf_mu = rmsetrain_trf[burnin: ].mean(axis=0)
    rmsetest_trf_mu = rmsetest_trf[burnin: ].mean(axis=0)


    train_rmse_trf[index-1] = rmsetrain_trf_mu
    test_rmse_trf[index-1] = rmsetest_trf_mu
    train_std_trf[index-1] = stdtrain_trf
    test_std_trf[index-1] = stdtest_trf

    # rmse and std for w/o transfer results
    rmsetrain = np.genfromtxt('synthetic_data '+str(index)+'/targettrainrmse.csv')
    rmsetest = np.genfromtxt('synthetic_data '+str(index)+'/targettestrmse.csv')
    burnin = int(0.1 * rmsetrain.shape[0])
    stdtrain = rmsetrain[burnin: ].std(axis=0)
    stdtest = rmsetest[burnin: ].std(axis=0)
    rmsetrain_mu = rmsetrain[burnin: ].mean(axis=0)
    rmsetest_mu = rmsetest[burnin: ].mean(axis=0)

    train_rmse[index-1] = rmsetrain_mu
    test_rmse[index-1] = rmsetest_mu
    train_std[index-1] = stdtrain
    test_std[index-1] = stdtest

n_groups = len(source_index)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.18
opacity = 0.8
capsize = 1


plt.bar(index + float(bar_width)/2, train_rmse, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=0.8, ecolor='C5', capsize=capsize),
                color = 'C0',
                yerr = train_std,
                label = 'train no-transfer')

plt.bar(index + float(bar_width)/2 + bar_width, train_rmse_trf, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=0.8, ecolor='C5', capsize=capsize),
                color = 'C2',
                yerr = train_std_trf,
                label = 'train full-transfer')


plt.bar(index + float(bar_width)/2 + 2 * bar_width, test_rmse, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=0.8, ecolor='C5', capsize=capsize),
                color = 'C1',
                yerr = test_std,
                label = 'test no-transfer')

plt.bar(index + float(bar_width)/2 + 3 * bar_width , test_rmse_trf, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=0.8, ecolor='C5', capsize=capsize),
                color = 'C3',
                yerr = test_std_trf,
                label = 'test full-transfer')

plt.title('Synthetic Data RMSE')
plt.xlabel('datasets')
plt.ylabel('RMSE')
plt.xticks(index+2*bar_width, source_index, rotation=0)
plt.legend()

plt.tight_layout()
plt.savefig('synthetic_data.png')
