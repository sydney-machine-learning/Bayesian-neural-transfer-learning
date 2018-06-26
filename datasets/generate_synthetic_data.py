import numpy as np
from sklearn.model_selection import train_test_split
import os

epsilon = np.random.normal(0, 1, 1)
delta = 0.2

x_target = np.abs(np.random.uniform(0, 1, 500 * 4).reshape((500,4)))
w_target = np.random.randn(4,1)
y_target = x_target.dot(w_target) + epsilon




x_source = np.abs(np.random.uniform(0, 1, 500 * 4).reshape((500,4)))
w_source = w_target + delta * np.random.randn(w_target.shape[0], w_target.shape[1])
y_source = x_source.dot(w_source) + epsilon



a = 0
b = 1

max_y = max(max(y_target), max(y_source))
min_y = min(min(y_target), min(y_source))

y_target = a + (y_target - min_y)*(b - a)/(max_y - min_y)
y_source = a + (y_source - min_y)*(b - a)/(max_y - min_y)


sourcedata = np.c_[x_source, y_source]

x_train, x_test, y_train, y_test  = train_test_split(x_target, y_target, test_size=0.9, random_state=45)
traindata = np.c_[x_train, y_train]
testdata = np.c_[x_test, y_test]

directory = './synthetic_data'

if not os.path.isdir(directory):
    os.mkdir(directory)

np.savetxt(directory+'/source.csv', sourcedata, delimiter=',')
np.savetxt(directory+'/targ0.2train.csv', traindata, delimiter=',')
np.savetxt(directory+'/target_test.csv', testdata, delimiter=',')
