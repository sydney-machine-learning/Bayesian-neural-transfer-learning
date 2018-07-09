import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import seaborn as sns
from scipy.stats import norm

mplt.style.use('bmh')

mplt.rcParams.update({'font.size': 10})
mplt.rc('xtick', labelsize=13)
mplt.rc('ytick', labelsize=13)

weights = np.genfromtxt('weights.csv', delimiter=',')
weights_trf = np.genfromtxt('weights_trf.csv', delimiter=',')

# def calc_posterior_analytical(data, x, mu_0, sigma_0):
#     sigma = 1.
#     n = len(data)
#     mu_post = (mu_0 / sigma_0**2 + data.sum() / sigma**2) / (1. / sigma_0**2 + n / sigma**2)
#     sigma_post = (1. / sigma_0**2 + n / sigma**2)**-1
#     return norm(mu_post, np.sqrt(sigma_post)).pdf(x)

for index in range(weights.shape[1]):
    ax = plt.subplot(111)
    x = np.linspace(-.5, .5, 500)
    plt.hist(weights_trf[400:,index], bins=50, alpha=0.5, facecolor='sandybrown', label='no-transfer', density=True)
    plt.hist(weights[400:, index], bins=50, alpha=0.5, facecolor='C0', label='transfer', density=True)
    plt.legend()
    plt.xlabel('Parameter value')
    plt.ylabel('Frequency')
    plt.savefig('weights/weight'+str(index+1)+'.png')
    plt.clf()
