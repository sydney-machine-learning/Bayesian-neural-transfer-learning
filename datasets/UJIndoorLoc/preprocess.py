import numpy as np

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


trainfile = 'trainingData.csv'
validationfile = 'validationData.csv'


traindata = np.genfromtxt(trainfile, delimiter=',', skip_header=1)
validationdata = np.genfromtxt(validationfile, delimiter=',', skip_header=1)

trainsize = traindata.shape[0]
valisize = validationdata.shape[0]

print trainsize, valisize

data = np.vstack((traindata, validationdata))
print data.shape

data = data[:, :-5]

data = normalizedata(data)

traindata = data[:trainsize, :]
validationdata = data[trainsize:, :]

print traindata.shape, validationdata.shape

datadict = {'trainingData':traindata, 'validationData':validationdata}

for file, data in datadict.items():
    building = {}
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
            # if file == 'trainingData':
            print file, building_id, floor_id, data.shape
