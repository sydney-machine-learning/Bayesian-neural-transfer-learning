import matplotlib.pyplot as plt
import numpy as np

building_id = [0, 1, 2]

train_rmse = np.zeros((len(building_id)))
test_rmse = np.zeros((len(building_id)))

train_rmse_trf = np.zeros((len(building_id)))
test_rmse_trf = np.zeros((len(building_id)))

train_std_trf = np.zeros((len(building_id)))
test_std_trf = np.zeros((len(building_id)))

train_std = np.zeros((len(building_id)))
test_std = np.zeros((len(building_id)))


for index in range(len(building_id)):
    
    building = building_id[index]
    rmsetrain_trf = np.genfromtxt('building'+str(building)+'/targettrftrainrmse.csv')
    rmsetest_trf = np.genfromtxt('building'+str(building)+'/targettrftestrmse.csv')
    
    burnin = int(0.1 * rmsetrain_trf.shape[0])
    stdtrain_trf = rmsetrain_trf[burnin: ].std(axis=0)
    stdtest_trf = rmsetest_trf[burnin: ].std(axis=0)
    rmsetrain_trf_mu = rmsetrain_trf[burnin: ].mean(axis=0)
    rmsetest_trf_mu = rmsetest_trf[burnin: ].mean(axis=0)

    
    train_rmse_trf[index] = rmsetrain_trf_mu
    test_rmse_trf[index] = rmsetest_trf_mu
    train_std_trf[index] = stdtrain_trf
    test_std_trf[index] = stdtest_trf
    
    # rmse and std for w/o transfer results
    rmsetrain = np.genfromtxt('building'+str(building)+'/targettrainrmse.csv')
    rmsetest = np.genfromtxt('building'+str(building)+'/targettestrmse.csv')
    burnin = int(0.1 * rmsetrain.shape[0])
    stdtrain = rmsetrain[burnin: ].std(axis=0)
    stdtest = rmsetest[burnin: ].std(axis=0)
    rmsetrain_mu = rmsetrain[burnin: ].mean(axis=0)
    rmsetest_mu = rmsetest[burnin: ].mean(axis=0)
    
    train_rmse[index] = rmsetrain_mu
    test_rmse[index] = rmsetest_mu
    train_std[index] = stdtrain
    test_std[index] = stdtest
    
#    print(rmsetest_mu, rmsetest_trf_mu, stdtest, stdtest_trf)

  

n_groups = len(building_id)
fig, ax = plt.subplots()  
index = np.arange(n_groups)
bar_width = 0.13
opacity = 0.8
capsize = 3
 
   
plt.bar(index + float(bar_width)/2, train_rmse, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C5', capsize=capsize),
                color = 'C0',
                yerr = train_std,
                label = 'train no-transfer')

plt.bar(index + float(bar_width)/2 + bar_width, train_rmse_trf, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C5', capsize=capsize),
                color = 'C2',
                yerr = train_std_trf,
                label = 'train full-transfer')
                
                
plt.bar(index + float(bar_width)/2 + 2 * bar_width, test_rmse, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C5', capsize=capsize),
                color = 'C1',
                yerr = test_std,
                label = 'test no-transfer')

plt.bar(index + float(bar_width)/2 + 3 * bar_width , test_rmse_trf, bar_width,
                alpha = opacity,
                error_kw = dict(elinewidth=1, ecolor='C5', capsize=capsize),
                color = 'C3',
                yerr = test_std_trf,
                label = 'test full-transfer')    
            
plt.xlabel('datasets')
plt.ylabel('RMSE')
plt.xticks(index+2*bar_width, building_id + [3], rotation=0)
plt.legend()

plt.tight_layout()
plt.savefig('barplt.png')

    
