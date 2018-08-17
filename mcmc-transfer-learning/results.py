import numpy as np

path = raw_input('Enter the path: ')
print('train_mu\tstdtrain\ttest_mu\t\tstdtest\t\ttrain_trf_mu\tstdtrain_trf\ttest_trf_mu\tstdtest_trf')

rmsetrain_trf = np.genfromtxt(path+'/joint_train_rmse.csv')
rmsetest_trf = np.genfromtxt(path+'/joint_test_rmse.csv')

burnin = int(0.1 * rmsetrain_trf.shape[0])
stdtrain_trf = rmsetrain_trf[burnin: ].std(axis=0)
stdtest_trf = rmsetest_trf[burnin: ].std(axis=0)
rmsetrain_trf_mu = rmsetrain_trf[burnin: ].mean(axis=0)
rmsetest_trf_mu = rmsetest_trf[burnin: ].mean(axis=0)


# rmse and std for w/o transfer results
rmsetrain = np.genfromtxt(path+'/target_train_rmse.csv')
rmsetest = np.genfromtxt(path+'/target_test_rmse.csv')
burnin = int(0.1 * rmsetrain.shape[0])
stdtrain = rmsetrain[burnin: ].std(axis=0)
stdtest = rmsetest[burnin: ].std(axis=0)
rmsetrain_mu = rmsetrain[burnin: ].mean(axis=0)
rmsetest_mu = rmsetest[burnin: ].mean(axis=0)

print('\t\t'.join(["%0.5f" % val for val in [rmsetrain_mu, stdtrain, rmsetest_mu, stdtest, rmsetrain_trf_mu, stdtrain_trf, rmsetest_trf_mu, stdtest_trf]]))
