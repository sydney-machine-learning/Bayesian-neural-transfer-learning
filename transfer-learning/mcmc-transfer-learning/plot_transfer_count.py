import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.style.use('bmh')

mpl.rcParams.update({'font.size': 10})
mpl.rc('xtick', labelsize=13)
mpl.rc('ytick', labelsize=13)



#prob = np.around(np.linspace(0.3, 1.0, 10), decimals=2)
prob = [0.3, 0.37, 0.45, 0.53, 0.61, 0.68, 0.76, 0.84, 0.92, 1.0]
#print(prob)

train_rmse_trf = np.zeros((len(prob)))
test_rmse_trf = np.zeros((len(prob)))

train_std_trf = np.zeros((len(prob)))
test_std_trf = np.zeros((len(prob)))

index = 0

for transfer_prob in prob:
    rmsetrain_trf = np.genfromtxt('./test_'+str(transfer_prob)+'/targettrftrainrmse.csv')
    rmsetest_trf = np.genfromtxt('./test_'+str(transfer_prob)+'/targettrftestrmse.csv')
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

transfer_cnt = np.genfromtxt('transfer_cnt.txt', delimiter=',')

fig, ax1 = plt.subplots()
# plt.errorbar(prob, train_rmse_trf, xerr=0, yerr=train_std_trf, color='C5')
plt.errorbar(prob, test_rmse_trf, xerr=0, yerr=test_std_trf, color='C4')
ax1.set_xlabel('transfer probabilities')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('RMSE', color='C4')
ax1.tick_params('y', colors='C4')

ax2 = ax1.twinx()
ax2.plot(prob, transfer_cnt, 'C7')
ax2.set_ylabel('transfer count', color='C7')
ax2.tick_params('y', colors='C7')

fig.tight_layout()
plt.savefig('transfer_prob.png')
