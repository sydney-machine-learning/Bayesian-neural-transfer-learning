import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.style.use('bmh')

mpl.rcParams.update({'font.size': 10})
mpl.rc('xtick', labelsize=13)
mpl.rc('ytick', labelsize=13)



#prob = np.around(np.linspace(0.3, 1.0, 10), decimals=2)
prob = [0.3, 0.37, 0.45, 0.61, 0.68, 0.76, 0.84, 0.92, 1.0]
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

fig, ax = plt.subplots()
opacity = 0.8
capsize = 3

plt.plot(prob, test_rmse_trf)
plt.plot(prob, train_rmse_trf)
plt.show()
#data = np.genfromtxt('transfer_cnt.txt', delimiter=',')
#x = np.around(np.linspace(0.3, 1.0, 10), decimals=2)
#plt.plot(x, data)
#plt.show()
