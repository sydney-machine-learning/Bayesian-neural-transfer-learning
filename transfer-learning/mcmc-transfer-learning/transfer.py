import numpy as np

file = open('results/wprop.csv', 'rb')
lines = file.read().split('\n')[:-1]

numSamples = len(lines)
w_size = len(lines[0].split(','))

weights = np.ones((numSamples, w_size))
burnin = int(0.1 * numSamples)


for index in range(len(lines)):
    line = lines[index]
    # print(index, line)
    w = np.array(list(map(float, line.split(','))))
    # print(w, index)
    weights[index, :] = w

weights = weights[burnin:, :]
w_mean = weights.mean(axis=0)
w_std = np.std(weights, axis=0)

w_prop = np.ones(w_mean.shape)

for index in range(w_mean.shape[0]):
    w_prop[index] = np.random.normal(w_mean[index], w_std[index], 1)

print(w_prop)

