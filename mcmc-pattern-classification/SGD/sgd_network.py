# !/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import multivariate_normal
from scipy.stats import norm
import math
import os
import sys
import pickle

sys.path.insert(0, '/home/arpit/Projects/Bayesian-neural-transfer-learning/preliminary/Iris/')
sys.path.insert(0, '/home/arpit/Projects/Bayesian-neural-transfer-learning/preliminary/WineQualityDataset/preprocess')

from preprocess import getdata

def convert_time(secs):
    if secs >= 60:
        mins = str(int(secs/60))
        secs = str(int(secs%60))
    else:
        secs = str(int(secs))
        mins = str(00)

    if len(mins) == 1:
        mins = '0'+mins

    if len(secs) == 1:
        secs = '0'+secs

    return [mins, secs]

# --------------------------------------------------------------------------

# An example of a class
class Network:
    def __init__(self, Topo, Train, Test, learn_rate = 0.5, alpha = 0.1):
        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
        np.random.seed(int(time.time()))
        self.lrate = learn_rate
        self.alpha = alpha
        self.NumSamples = self.TrainData.shape[0]

        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer
        # self.W3 = np.random.randn(self.Top[2], self.Top[3]) / np.sqrt(self.Top[2])
        # self.B3 = np.random.randn(1, self.Top[3]) / np.sqrt(self.Top[2])  # bias second layer



        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        # self.hidout2 = np.zeros((1,self.Top[2]))
        self.out = np.zeros((1, self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.Top[2]
        return sqerror

    def sampleAD(self, actualout):
        error = np.subtract(self.out, actualout)
        moderror = np.sum(np.abs(error)) / self.Top[2]
        return moderror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer
        # z3 = self.hidout2.dot(self.W3) - self.B3
        # self.out = self.sigmoid(z3)

    def BackwardPass(self, Input, desired):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        # hid_delta2 = out_delta.dot(self.W3.T) * (self.hidout2 * (1 - self.hidout2))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        # self.W3 += (self.hidout2.T.dot(out_delta) * self.lrate)
        # self.B3 += (-1 * self.lrate * out_delta)
        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)

        # layer = 1  # hidden to output
        # for x in xrange(0, self.Top[layer]):
        #     for y in xrange(0, self.Top[layer + 1]):
        #         self.W2[x, y] += self.lrate * out_delta[y] * self.hidout[x]
        # for y in xrange(0, self.Top[layer + 1]):
        #     self.B2[y] += -1 * self.lrate * out_delta[y]
        #
        # layer = 0  # Input to Hidden
        # for x in xrange(0, self.Top[layer]):
        #     for y in xrange(0, self.Top[layer + 1]):
        #         self.W1[x, y] += self.lrate * hid_delta[y] * Input[x]
        # for y in xrange(0, self.Top[layer + 1]):
        #     self.B1[y] += -1 * self.lrate * hid_delta[y]

    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]
        # w_layer3size = self.Top[2] * self.Top[3]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))

        # w_layer3 = w[w_layer2size:w_layer3size + w_layer2size]
        # self.W3 = np.reshape(w_layer3, (self.Top[2], self.Top[3]))

        self.B1 = w[w_layer1size + w_layer2size :w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1] :w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]
        # self.B3 = w[w_layer1size + w_layer2size + w_layer3size + self.Top[1] + self.Top[2]:w_layer1size + w_layer2size + w_layer3size + self.Top[1] + self.Top[2] + self.Top[3]]


    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        # w3 = self.W3.ravel()
        w = np.concatenate([w1, w2, self.B1, self.B2])
        return w

    def langevin_gradient(self, data, w, depth):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for i in xrange(0, depth):
            for i in xrange(0, size):
                pat = i
                Input = data[pat, 0:self.Top[0]]
                Desired = data[pat, self.Top[0]:]
                self.ForwardPass(Input)
                self.BackwardPass(Input, Desired)

        w_updated = self.encode()

        return  w_updated

    def evaluate_proposal(self, data, w ):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros((size,self.Top[2]))

        for i in xrange(0, size):  # to see what fx is produced by your current weight update
            Input = data[i, 0:self.Top[0]]
            self.ForwardPass(Input)
            fx[i] = self.out

        return fx

    def TestNetwork(self, phase, erTolerance):
        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        Output = np.zeros((1, self.Top[2]))
        if phase == 1:
            Data = self.TestData
        if phase == 0:
            Data = self.TrainData
        clasPerf = 0
        sse = 0
        testSize = Data.shape[0]
        self.W1 = self.BestW1
        self.W2 = self.BestW2  # load best knowledge
        # self.W3 = self.BestW3
        self.B1 = self.BestB1
        self.B2 = self.BestB2  # load best knowledge
        # self.B3 = self.BestB3


        for s in xrange(0, testSize):

            Input[:] = Data[s, 0:self.Top[0]]
            Desired[:] = Data[s, self.Top[0]:]

            self.ForwardPass(Input)
            sse = sse + self.sampleEr(Desired)

            if (np.isclose(self.out, Desired, atol=erTolerance).all()):
                clasPerf = clasPerf + 1

        return (sse / testSize, float(clasPerf) / testSize * 100)

    def saveKnowledge(self):
        self.BestW1 = self.W1
        self.BestW2 = self.W2
        # self.BestW3 = self.W3
        self.BestB1 = self.B1
        self.BestB2 = self.B2
        # self.BestB3 = self.B3

    def plot_err(self, mse_lis, mad_lis, depth):

        ax = plt.subplot(111)
        plt.plot(range(depth), mse_lis, label="mse")
        plt.plot(range(depth), mad_lis, label="mad")

        leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)

        plt.xlabel('Epoch')
        plt.ylabel('Mean Error')
        plt.title(' Mean Error')
        plt.savefig('iris.png')
        plt.clf()

    def BP_GD(self, stocastic, vanilla, depth):  # BP with SGD (Stocastic BP)
        # self.momenRate = mRate
        # self.NumSamples = numSamples

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        # Er = []#np.zeros((1, self.Max))
        epoch = 0
        bestmse = 100
        bestmad = 100
        bestTrain = 0

        mad_lis = []
        mse_lis = []

        train_acc = []
        test_acc = []
        train_sse = []
        test_sse = []

        start = time.time()
        # while  epoch < self.Max and bestTrain < self.minPerf :
        while epoch < depth:
            sse = 0
            sad = 0
            for s in xrange(0, self.NumSamples):

                if (stocastic):
                    pat = random.randint(0, self.NumSamples - 1)
                else:
                    pat = s

                Input[:] = self.TrainData[pat, :self.Top[0]]
                Desired[:] = self.TrainData[pat, self.Top[0]:]

                self.ForwardPass(Input)
                self.BackwardPass(Input, Desired)
                sse = sse + self.sampleEr(Desired)
                sad = sad + self.sampleAD(Desired)

            mse = np.sqrt(sse / self.NumSamples * self.Top[2])
            mad = sad / self.NumSamples * self.Top[2]

            if mse < bestmse:
                bestmse = mse
                self.saveKnowledge()
                (x, bestTrain) = self.TestNetwork(0, 0.2)

            if mad < bestmad:
                bestmad = mad

            mse_lis.append(bestmse)
            mad_lis.append(bestmad)

            if epoch%10 == 0:
                sse, acc = self.TestNetwork(phase=1, erTolerance=etol)
                test_acc.append(acc)
                test_sse.append(sse)

                sse, acc = self.TestNetwork(phase=0, erTolerance=etol_tr)
                train_acc.append(acc)
                train_sse.append(sse)

            elapsed = convert_time(time.time() - start)
            # Er = np.append(Er, mse)
            sys.stdout.write('\rEpoch: '+str(epoch+1)+"/"+ str(depth)+" mse: "+ str(mse) + " mad: " + str(mad) + " Time elapsed: "+str(elapsed[0])+":"+str(elapsed[1]))

            epoch = epoch + 1

        # self.plot_err(mse_lis, mad_lis, depth)

        # return (mse, mad, bestmse, bestTrain, epoch)
        return train_acc, test_acc

    def transfer_weights(self, source):
        self.W1 = source.W1
        self.W2 = source.W2
        # self.W3 = source.W3
        self.B1 = source.B1
        self.B2 = source.B2
        # self.B3 = source.B3


# --------------------------------------------------------------------------




# --------------------------------------------------------------------------


def pickle_knowledge(obj, pickle_file):
    pickling_on = open(pickle_file, 'wb')
    pickle.dump(obj, pickling_on)
    pickling_on.close()

# --------------------------------------------------------------------------
if __name__ == '__main__':
    input = 11
    hidden = 94
    output = 4
    topo = [input, hidden, output]
    # print(traindata.shape, testdata.shape)
    lrate = 0.67
    etol_tr = 0.2
    etol = 0.6
    alpha = 0.1

    minepoch = 0
    maxepoch = 500

    #
    # traindata, testdata = getdata('WineQualityDataset/winequality-white.csv')
    # y_train = traindata[:, input:]
    # y_test = testdata[:, input:]

    # # Source Dataset Network
    # network_white = Network(Topo=topo, Train=traindata, Test=testdata, learn_rate =lrate, alpha=alpha)
    # print("\nWine Quality White (Source):")
    # train_acc, test_acc = network_white.BP_GD(stocastic=False, vanilla=1, depth=1500)
    #
    # print "\nTrain Data performance: "
    # sse, acc = network_white.TestNetwork(phase=0, erTolerance=etol_tr)
    # print("sse: "+ str(sse) + " acc: "+ str(acc))
    # print "Test Data performance: "
    # sse, acc = network_white.TestNetwork(phase=1, erTolerance=etol)
    # print("sse: " + str(sse) + " acc: " + str(acc))
    #
    # pickle_knowledge(network_white, 'winequality-white-knowledge.pickle')

    pickler = open('winequality-white-knowledge.pickle', 'rb')
    network_white = pickle.load(pickler)


    # Target network without transfer
    traindata, testdata = getdata('WineQualityDataset/winequality-red.csv')
    network_red = Network(Topo=topo, Train=traindata, Test=testdata, learn_rate=lrate, alpha=alpha)
    print("\nWine Quality Red Without Transfer:")
    train_acc_wo, test_acc_wo = network_red.BP_GD(stocastic=True, vanilla=1, depth=maxepoch)
    print "\nTrain Data performance: "
    sse, acc = network_red.TestNetwork(phase=0, erTolerance=etol_tr)
    print("sse: "+ str(sse) + " acc: "+ str(acc))
    print "Test Data performance: "
    sse, acc = network_red.TestNetwork(phase=1, erTolerance=etol)
    print("sse: " + str(sse) + " acc: " + str(acc))
    pickle_knowledge(network_red, 'winequality-red-knowledge.pickle')



    # Transfer knowledge from source network to the target network
    network_red_trf = Network(Topo=topo, Train=traindata, Test=testdata, learn_rate =lrate, alpha=alpha)
    network_red_trf.transfer_weights(network_white)
    print("\nWine Quality Red With Transfer:")
    train_acc_w, test_acc_w = network_red_trf.BP_GD(stocastic=True, vanilla=1, depth=maxepoch)

    print "\nTrain Data performance: "
    sse, acc = network_red_trf.TestNetwork(phase=0, erTolerance=etol_tr)
    print("sse: "+ str(sse) + " acc: "+ str(acc))
    print "\nTest Data performance: "
    sse, acc = network_red_trf.TestNetwork(phase=1, erTolerance=etol)
    print("sse: " + str(sse) + " acc: " + str(acc))

    pickle_knowledge(network_red_trf, 'winequality-red-knowledge-trf.pickle')



    ax = plt.subplot(111)
    plt.plot(range(0, 500, 10), train_acc_w, label="train_w_trf")
    plt.plot(range(0, 500, 10), train_acc_wo, label="train_wo_trf")


    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Wine Quality-Red Accuracy plot')
    plt.savefig('accuracy-red-train.png')

    plt.clf()

    plt.plot(range(0, 500, 10), test_acc_w, label="test_w_trf")
    plt.plot(range(0, 500, 10), test_acc_wo, label="test_wo_trf")
    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Wine Quality-Red Accuracy plot')
    plt.savefig('accuracy-red-test.png')