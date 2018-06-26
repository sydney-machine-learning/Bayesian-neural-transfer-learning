import numpy as np
from sklearn.model_selection import train_test_split
import os

epsilon = np.random.normal(0, 1, 1)
delta = 1

x_target = np.abs(np.random.randn(500,4))
w_target = np.random.randn(4,1)
y_target = x_target.dot(w_target) + epsilon


x_source = np.abs(np.random.randn(500,4))
w_source = w_target + delta * np.random.randn(w_target.shape[0], w_target.shape[1])
y_source = x_source.dot(w_source) + epsilon


sourcedata = np.c_[x_source, y_source]

x_train, x_test, y_train, y_test  = train_test_split(x_target, y_target, test_size=0.9, random_state=45)
traindata = np.c_[x_train, y_train]
testdata = np.c_[x_test, y_test]

directory = './synthetic_data'

if not os.path.isdir(directory):
    os.mkdir(directory)

np.savetxt(directory+'/source.csv', sourcedata, delimiter=',')
np.savetxt(directory+'/target_train.csv', traindata, delimiter=',')
np.savetxt(directory+'/target_test.csv', testdata, delimiter=',')