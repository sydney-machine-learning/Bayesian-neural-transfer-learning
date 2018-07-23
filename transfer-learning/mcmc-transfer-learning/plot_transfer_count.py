import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.style.use('bmh')

mpl.rcParams.update({'font.size': 12})
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=15)


prob = np.around(np.linspace(0.3, 1.2, 20), decimals=2)[:-5]
print(prob)
#print(prob)

train_rmse_trf = np.zeros((len(prob)))
test_rmse_trf = np.zeros((len(prob)))

train_std_trf = np.zeros((len(prob)))
test_std_trf = np.zeros((len(prob)))

index = 0

for transfer_prob in prob:
        rmsetrain_trf = np.genfromtxt('transfer_prob/test_'+str(transfer_prob)+'/targettrftrainrmse.csv')
        rmsetest_trf = np.genfromtxt('transfer_prob/test_'+str(transfer_prob)+'/targettrftestrmse.csv')
        burnin = int(0.1 * rmsetrain_trf.shape[0])
        stdtrain_trf = rmsetrain_trf[burnin: ].std(axis=0)
        stdtest_trf = rmsetest_trf[burnin: ].std(axis=0)
        rmsetrain_trf_mu = rmsetrain_trf[burnin: ].mean(axis=0)
        rmsetest_trf_mu = rmsetest_trf[burnin: ].mean(axis=0)


        train_rmse_trf[index] = rmsetrain_trf_mu
        test_rmse_trf[index] = rmsetest_trf_mu
        train_std_trf[index] = stdtrain_trf
        test_std_trf[index] = stdtest_trf

        index += 1

transfer_cnt = np.genfromtxt('transfer_prob.txt', delimiter=',')[: -5]

fig, ax1 = plt.subplots()
# plt.errorbar(prob, train_rmse_trf, xerr=0, yerr=train_std_trf, color='C5')
plt.errorbar(prob, test_rmse_trf, xerr=0, yerr=test_std_trf, color='C4')
ax1.set_xlabel('transfer probabilities')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Test RMSE', color='C4')
ax1.tick_params('y', colors='C4')

ax2 = ax1.twinx()
ax2.plot(prob, transfer_cnt, 'C3')
ax2.set_ylabel('transfer count', color='C3')
ax2.tick_params('y', colors='C3')

fig.tight_layout()
# plt.show()
plt.savefig('transfer_prob.png')
