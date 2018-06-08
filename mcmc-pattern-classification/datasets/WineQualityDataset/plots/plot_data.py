import matplotlib.pyplot as plt
from collections import Counter
from numpy import genfromtxt

path = 'WineQualityDataset/'
wine = ['winequality-white', 'winequality-red']

for data in wine:
    fig = plt.figure()
    file = path+data+'.csv'
    my_data = genfromtxt(file, delimiter=';')
    my_data = my_data[1:, -1].astype(int)
    
    cnt = dict(Counter(my_data))
    keys, values = zip(*cnt.items())

    print(cnt)
    print(zip(keys,values))


    plt.bar(keys, values)
    plt.savefig(data+'.png')
