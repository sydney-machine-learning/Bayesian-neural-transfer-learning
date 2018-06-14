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
    features[0,index] = normalize(x, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(features[0, index], labels[0, index], test_size = 0.25, random_state = 0)

    traindata = np.c_[X_train,y_train]
    testdata = np.c_[X_test, y_test]

    os.mkdir('tasks/task'+str(index+1))

    with open('tasks/task'+str(index+1)+'/train.csv', 'w') as target:
         np.savetxt(target, traindata, delimiter=',')

    with open('tasks/task'+str(index+1)+'/test.csv', 'w') as target:
         np.savetxt(target, testdata, delimiter=',')
