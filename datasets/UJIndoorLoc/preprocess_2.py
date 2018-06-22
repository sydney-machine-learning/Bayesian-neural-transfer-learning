import numpy as np
from sklearn.model_selection import train_test_split


def normalizedata(data):
    a = 0
    b = 1

    longmax = -7299.786516730871000
    longmin = -7695.9387549299299000
    latmin = 4864745.7450159714
    latmax = 4865017.3646842018

    long = data[:, 520]
    lat = data[:, 521]

    long = np.ones(long.shape)*a + (long - np.ones(long.shape)*longmin)*(b - a)/(longmax - longmin)
    lat = a + (lat - latmin)*(b - a)/(latmax - latmin)

    data[:, 520] = long
    data[:, 521] = lat

    return data


sourcefile = 'trainingData.csv'
targetfile = 'validationData.csv'

building = {}
sourcedata = np.genfromtxt(sourcefile, delimiter=',', skip_header=1)
targetdata = np.genfromtxt(targetfile, delimiter=',', skip_header=1)

sourcesize = sourcedata.shape[0]
targetsize = targetdata.shape[0]

# print sourcesize, targetsize

data = np.vstack((sourcedata, targetdata))
# print data.shape

data = data[:, :-5]

data = normalizedata(data)

sourcedata = data[:sourcesize, :]
targetdata = data[sourcesize:, :]

# print sourcedata.shape, targetdata.shape

datadict = {'sourceData':sourcedata, 'targetData':targetdata}

for file, data in datadict.items():
    building = {}
    for index in range(data.shape[0]):
        building_id = int(data[index, -1])
        try:
            building[building_id].append(data[index, :])
        except Exception as e:
            building[building_id] = [data[index, :]]
    for building_id, data in building.items():
        data = np.array(data)
        print data.shape
        X = data[:, :-4]
        y = data[:, -4:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        traindata = np.c_[X_train, y_train]
        testdata = np.c_[X_test, y_test]
        np.savetxt(file+str(building_id)+'train.csv', traindata, delimiter=',')
        np.savetxt(file+str(building_id)+'test.csv', testdata, delimiter=',')
