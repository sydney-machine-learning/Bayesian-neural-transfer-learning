import scipy.io
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt


def normalise(data):
    sc = MinMaxScaler(copy=True, feature_range=(0,1))
    data = sc.fit_transform(data)
    data = normalize(data, norm='l2')
    ax = plt.subplot(111)
    x = np.array(np.arange(100))
    joints = range(7)
    for i in range(10):
        plt.plot(x, data[i*100:(i+1)*100, 21+3], '.' , label='f'+str(3))
        plt.plot(x, data[i*100:(i+1)*100, 21+6], '.' , label='f'+str(6))
        plt.legend()
        plt.xlabel('record')
        plt.ylabel('y')
        plt.title('fx')
        plt.savefig(str(i)+'.png')
        plt.clf()
    return data

def getdata(source=3, target=6):
    dataset_train = scipy.io.loadmat('sarcos_inv.mat')
    dataset_test = scipy.io.loadmat('sarcos_inv_test.mat')

    np.savetxt('sarcos_inv.csv', dataset_train['sarcos_inv'], delimiter=',')
    np.savetxt('sarcos_inv_test.csv', dataset_test['sarcos_inv_test'], delimiter=',')

    dataset_train = dataset_train['sarcos_inv']
    dataset_test = dataset_test['sarcos_inv_test']

    dataset = np.vstack([dataset_train, dataset_test])
    dataset = normalise(dataset)

    x = dataset[:, :21]
    source_y = dataset[:, 21+source]
    target_y= dataset[:, 21+target]

    source = np.c_[x, source_y]
    X_train, X_test, y_train, y_test = train_test_split(x, target_y, test_size = 0.95, random_state = int(time.time()))

    target_train = np.c_[X_train, y_train]
    target_test = np.c_[X_test, y_test]

    return source, target_train, target_test

if __name__ == '__main__':
    source, target_train, target_test = getdata(source=6, target=3)
    np.savetxt('source.csv', source, delimiter=',')
    np.savetxt('target_train.csv', target_train, delimiter=',')
    np.savetxt('target_test.csv', target_test, delimiter=',')
