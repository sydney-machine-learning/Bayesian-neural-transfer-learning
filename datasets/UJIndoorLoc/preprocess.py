import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

trainfile = 'trainingData.csv'
validationfile = 'validationData.csv'

building = {}
traindata = np.genfromtxt(trainfile, delimiter=',', skip_header=1)
validationdata = np.genfromtxt(validationfile, delimiter=',', skip_header=1)

trainsize = traindata.shape[0]
valisize = validationdata.shape[0]

print trainsize, valisize

data = np.vstack((traindata, validationdata))
print data.shape

data = data[:, :-5]

min_max_scaler = MinMaxScaler()
data_scl = min_max_scaler.fit_transform(data[:, :-2])
data[:, :-2] = normalize(data_scl, norm='l2')



x = data[:, :520]
y = data[:, 520:]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

traindata = np.c_[X_train, y_train]
validationdata = np.c_[X_test, y_test]

print traindata.shape, validationdata.shape

datadict = {'trainingData':traindata, 'validationData':validationdata}

for file, data in datadict.items():
    for index in range(data.shape[0]):
        building_id = int(data[index, -1])
        try:
            building[building_id].append(data[index, :])
        except Exception as e:
            building[building_id] = [data[index, :]]

    for building_id,data in building.items():
        floor = {}
        data = np.array(building[building_id])
        for index in range(data.shape[0]):
            floor_id = int(data[index, -2])
            try:
                floor[floor_id].append(data[index, :])
            except:
                floor[floor_id] = [data[index, :]]
        for floor_id in floor.keys():
            floor[floor_id] = np.array(floor[floor_id])
        building[building_id] = floor

    for building_id, dict in building.items():
        for floor_id, data in building[building_id].items():
            np.savetxt(file+'/'+''.join([str(building_id),str(floor_id)])+'.csv', data, delimiter=',')
