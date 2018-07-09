import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
mplt.style.use('bmh')

mplt.rcParams.update({'font.size': 10})
mplt.rc('xtick', labelsize=13)
mplt.rc('ytick', labelsize=13)

weights = np.genfromtxt('weights.csv', delimiter=',')
weights_trf = np.genfromtxt('weights_trf.csv', delimiter=',')
print(weights)
#
# plt.hist(weights, bins=50, alpha=0.5, facecolor='sandybrown')
# plt.xlabel('Parameter value')
# plt.ylabel('Frequency')
# plt.savefig('weight.png')
# plt.clf()
#
# plt.hist(weights_trf, bins = 50, alpha=0.5, facecolor='sandybrown')
# plt.xlabel('Parameter value')
# plt.ylabel('Frequency')
# plt.savefig('weight_trf.png')
# plt.clf()

for index in range(weights.shape[1]):
    ax = plt.subplot(111)
    plt.hist(weights_trf[:,index], bins=50, alpha=0.5, facecolor='sandybrown', label='no-transfer')
    plt.hist(weights[:, index], bins=50, alpha=0.5, facecolor='C0', label='transfer')
    plt.legend()
    plt.xlabel('Parameter value')
    plt.ylabel('Frequency')
    plt.savefig('weights/weight'+str(index+1)+'.png')
    plt.clf()
