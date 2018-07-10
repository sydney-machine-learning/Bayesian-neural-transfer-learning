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
#
# def calc_posterior_analytical(data, x, mu_0, sigma_0):
#     sigma = 1
#     n = len(data)
#     mu_post = (mu_0 / sigma_0**2 + data.sum() / sigma**2) / (1. / sigma_0**2 + n / sigma**2)
#     sigma_post = (1. / sigma_0**2 + n / sigma**2)**-1
#     return norm(mu_post, np.sqrt(sigma_post)).pdf(x)

for index in range(weights.shape[1]-2):
    ax = plt.subplot(111)
    x = np.linspace(-.5, .5, 500)
    plt.hist(weights[400:, index], bins=16, alpha=0.7, facecolor='sandybrown', edgecolor='r', label='source', density=True)
    plt.hist(weights[400:, -2], bins=20, alpha=0.7, facecolor='C0', edgecolor='b', label='target (no-transfer)', density=True)
    plt.hist(weights[400:, -1], bins=30, alpha=0.7, facecolor='C8', edgecolor='g', label='target (transfer)', density=True)
    plt.legend()
    plt.title('Weight Density plot Source vs Target')
    plt.xlabel('Parameter value')
    plt.ylabel('Density')
    plt.savefig('weight'+str(index+1)+'.png')
    plt.clf()
