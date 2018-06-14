import scipy.io
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


dataset = scipy.io.loadmat('LandmineData.mat')
features = dataset['feature']
labels = dataset['label']

for index in range(dataset['feature'].shape[1]):
    sc_X = StandardScaler()
    x = sc_X.fit_transform(features[0,index])
    x = normalize(x, norm='l2')

    col = np.zeros((labels[0, index].shape[0], 1))
    y = labels[0, index]

    for itr in range(y.shape[0]):
        if y[itr, 0] == 1:
            col[itr, 0] = 0
        else:
            col[itr, 0] = 1

    y = np.c_[y, col]


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

    traindata = np.c_[X_train,y_train]
    testdata = np.c_[X_test, y_test]

    with open('tasks/task'+str(index+1)+'/train.csv', 'w') as target:
         np.savetxt(target, traindata, delimiter=',')

    with open('tasks/task'+str(index+1)+'/test.csv', 'w') as target:
         np.savetxt(target, testdata, delimiter=',')
