import numpy as np
from sklearn.model_selection import train_test_split
import time
import os

def normalizedata(data):
    a = 0
    b = 1

    longmax = -7299.786516730871000
    longmin = -7695.9387549299299000
    latmin = 4864745.7450159714
    latmax = 4865017.3646842018

    longi = data[:, 520].copy()
    lat = data[:, 521].copy()

    long_sc = np.ones(longi.shape)*a + (longi - np.ones(longi.shape)*longmin)*(b - a)/(longmax - longmin)
    lat_sc = a + (lat - latmin)*(b - a)/(latmax - latmin)

    data[:, 520] = long_sc
    data[:, 521] = lat_sc

    return data, longi, lat


sourcefile = 'trainingData.csv'
targetfile = 'validationData.csv'

sourcedata = np.genfromtxt(sourcefile, delimiter=',', skip_header=1)
targetdata = np.genfromtxt(targetfile, delimiter=',', skip_header=1)

sourcesize = sourcedata.shape[0]
targetsize = targetdata.shape[0]

# print sourcesize, targetsize

data = np.vstack((sourcedata, targetdata))
# print data.shape

data = data[:, :-5]

data, longi, lat = normalizedata(data.copy())

data = np.c_[data, longi, lat]

sourcedata = data[:sourcesize, :]
targetdata = data[sourcesize:, :]

# print sourcedata.shape, targetdata.shape

datadict = {'sourceData':sourcedata, 'targetData':targetdata}
sizedict = {'sourceData':0.05, 'targetData':0.95}

for file, data in datadict.items():
    if not os.path.isdir(file):
        os.mkdir(file)
    building = {}
    for index in range(data.shape[0]):
        building_id=int(data[index, -3])
        try:
            building[building_id].append(data[index, :])
        except Exception as e:
            building[building_id] = [data[index, :]]
    for building_id, data in building.items():
        data = np.array(data)
        data = np.delete(data, 523, axis=1)
        data = np.delete(data, 522, axis=1)
        X = data[:, :-4]
        y = data[:, -4:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sizedict[file], random_state=int(time.time()))
        traindata = np.c_[X_train, y_train]
        testdata = np.c_[X_test, y_test]
        # np.savetxt(file+'/train.csv', traindata, delimiter=',')
        # np.savetxt(file+'/test.csv', testdata, delimiter=',')
        np.savetxt(file+'/'+str(building_id)+'train.csv', traindata, delimiter=',')
        np.savetxt(file+'/'+str(building_id)+'test.csv', testdata, delimiter=',')
        print(traindata.shape, testdata.shape)
