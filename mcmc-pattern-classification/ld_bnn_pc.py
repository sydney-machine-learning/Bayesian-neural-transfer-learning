# i/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import multivariate_normal
from scipy.stats import norm
import math
import os
import sys


# sys.path.insert(0, '/home/arpit/Projects/Bayesian-neural-transfer-learning/preliminary/WineQualityDataset/preprocess/')
# from preprocess import getdata


# An example of a class
class Network:
    def __init__(self, Topo, Train, Test, learn_rate):
        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
        np.random.seed()
        self.lrate = learn_rate

        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer

        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.Top[2]
        return sqerror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        #self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        #self.B2 += (-1 * self.lrate * out_delta)
        #self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        #self.B1 += (-1 * self.lrate * hid_delta)

        layer = 1  # hidden to output
        for x in xrange(0, self.Top[layer]):
            for y in xrange(0, self.Top[layer + 1]):
                self.W2[x, y] += self.lrate * out_delta[y] * self.hidout[x]
        for y in xrange(0, self.Top[layer + 1]):
            self.B2[y] += -1 * self.lrate * out_delta[y]

        layer = 0  # Input to Hidden
        for x in xrange(0, self.Top[layer]):
            for y in xrange(0, self.Top[layer + 1]):
                self.W1[x, y] += self.lrate * hid_delta[y] * Input[x]
        for y in xrange(0, self.Top[layer + 1]):
            self.B1[y] += -1 * self.lrate * hid_delta[y]

    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]


    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
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



# --------------------------------------------------------------------------

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


# -------------------------------------------------------------------


class MCMC:
    def __init__(self, samples, traindata, testdata, topology):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        self.wsize = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        # ----------------

    def softmax(self, fx):
        ex = np.exp(fx)
        sum_ex = np.sum(ex, axis = 1)
        sum_ex = np.multiply(np.ones(ex.shape), sum_ex[:, np.newaxis])
        prob = np.divide(ex, sum_ex)
        return prob

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, data, w):
        y = data[:, self.topology[0]:]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.rmse(fx, y)
        prob = self.softmax(fx)
        loss = 0
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i, j] == 1:
                    loss += np.log(prob[i, j] + 0.0001)

        out = np.argmax(fx, axis=1)
        y_out = np.argmax(y, axis=1)
        count = 0
        for i in range(y_out.shape[0]):
            if out[i] == y_out[i]:
                count += 1
        acc = float(count)/y_out.shape[0] * 100
        # acc = 0
        return [loss, fx, rmse, acc]


    def prior_likelihood(self, sigma_squared, w):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2
        return log_loss


    def genweights(self, w_mean, w_std):
        w_prop = np.ones(w_mean.shape)
        for index in range(w_mean.shape[0]):
            w_prop[index] = np.random.normal(w_mean[index], w_std[index], 1)
        return w_prop


    def transfer(self, stdmulconst):
        file = open('knowledge/wprop.csv', 'rb')
        lines = file.read().split('\n')[:-1]

        weights = np.ones((self.samples, self.wsize))
        burnin = int(0.1 * self.samples)

        for index in range(len(lines)):
            line = lines[index]
            w = np.array(list(map(float, line.split(','))))
            # print(w.shape)
            weights[index, :] = w

        weights = weights[burnin:, :]
        w_mean = weights.mean(axis=0)
        w_std = stdmulconst * np.std(weights, axis=0)
        return self.genweights(w_mean, w_std)


    def sampler(self, w_pretrain, w_limit, transfer = False):

        # Create file objects to write the attributes of the samples
        trainaccfile = open('./knowledge/trainacc.csv', 'w')
        testaccfile = open('./knowledge/testacc.csv', 'w')

        trainrmsefile = open('./knowledge/trainrmse.csv', 'w')
        testrmsefile = open('./knowledge/testrmse.csv', 'w')


        # ------------------- initialize MCMC

        start = time.time()
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples

        self.sgd_depth = 1

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)



        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]:]
        y_train = self.traindata[:, netw[0]:]

        pos_w = np.ones((self.wsize, ))  # posterior of all weights and bias over all samples


        w = w_pretrain

        w_proposal = w_pretrain

        step_w = w_limit  # defines how much variation you need in changes to w
        learn_rate = 0.67

        # --------------------- Declare FNN and initialize

        neuralnet = Network(self.topology, self.traindata, self.testdata, learn_rate)

        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)


        sigma_squared = 25

        sigma_diagmat = np.zeros((self.wsize, self.wsize))
        np.fill_diagonal(sigma_diagmat, step_w)


        prior = self.prior_likelihood(sigma_squared, w)  # takes care of the gradients

        [likelihood, pred_train, rmsetrain, trainacc] = self.likelihood_func(neuralnet, self.traindata, w)
        [likelihood_ignore, pred_test, rmsetest, testacc] = self.likelihood_func(neuralnet, self.testdata, w)




        if transfer:
            np.reshape(w_proposal, (1, w_proposal.shape[0]))
            with open('./knowledge/wprop.csv', 'w') as wprofile:
                np.savetxt(wprofile, [w_proposal], delimiter=',', fmt='%.5f')

        np.savetxt(trainaccfile, [trainacc])
        np.savetxt(testaccfile, [testacc])
        np.savetxt(trainrmsefile, [rmsetrain])
        np.savetxt(testrmsefile, [rmsetest])


        trainacc_prev = trainacc
        testacc_prev = testacc
        rmsetest_prev = rmsetest
        rmsetrain_prev = rmsetrain
        wpro_prev = w_proposal

        naccept = 0
        # print 'begin sampling using mcmc random walk'

        for i in range(samples - 1):

            w_gd = neuralnet.langevin_gradient(self.traindata, w.copy(), self.sgd_depth)  # Eq 8

            w_proposal = w_gd + np.random.normal(0, step_w, self.wsize) # Eq 7

            w_prop_gd = neuralnet.langevin_gradient(self.traindata, w_proposal.copy(), self.sgd_depth)

            diff_prop = np.log(multivariate_normal.pdf(w, w_prop_gd, sigma_diagmat) - np.log(
                multivariate_normal.pdf(w_proposal, w_gd, sigma_diagmat)))

            [likelihood_proposal, pred_train, rmsetrain, trainacc] = self.likelihood_func(neuralnet, self.traindata, w_proposal)
            [likelihood_ignore, pred_test, rmsetest, testacc] = self.likelihood_func(neuralnet, self.testdata, w_proposal)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, w_proposal)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_prior = prior_prop - prior

            mh_prob = min(700, (diff_likelihood + diff_prior + diff_prop))
            mh_prob = min(1, math.exp(mh_prob))

            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                naccept += 1
                likelihood = likelihood_proposal
                prior = prior_prop
                w = w_proposal

                # print i, trainacc, rmsetrain
                elapsed = convert_time(time.time() - start)
                sys.stdout.write(
                    '\rSamples: ' + str(i + 2) + "/" + str(samples)
                    + "\tTrain accuracy: " + str(trainacc)
                    + " Train RMSE: "+ str(rmsetrain)
                    + "\tTest accuracy: " + str(testacc)
                    + " Test RMSE: " + str(rmsetest)
                    + "\tTime elapsed: " + str(elapsed[0]) + ":" + str(elapsed[1]) )

                # save arrays to file
                if transfer:
                    np.reshape(w_proposal, (1, w_proposal.shape[0]))
                    with open('./knowledge/wprop.csv', 'a') as wprofile:
                        np.savetxt(wprofile, [w_proposal], delimiter=',', fmt='%.5f')

                np.savetxt(trainaccfile, [trainacc])
                np.savetxt(testaccfile, [testacc])
                np.savetxt(trainrmsefile, [rmsetrain])
                np.savetxt(testrmsefile, [rmsetest])

                #save values into previous variables
                wpro_prev = w_proposal
                trainacc_prev = trainacc
                testacc_prev = testacc
                rmsetrain_prev = rmsetrain
                rmsetest_prev = rmsetest

            else:
                if transfer:
                    np.reshape(wpro_prev, (1, wpro_prev.shape[0]))
                    with open('./knowledge/wprop.csv', 'a') as wprofile:
                        np.savetxt(wprofile, [wpro_prev], delimiter=',', fmt='%.5f')
                np.savetxt(trainaccfile, [trainacc_prev])
                np.savetxt(testaccfile, [testacc_prev])
                np.savetxt(trainrmsefile, [rmsetrain_prev])
                np.savetxt(testrmsefile, [rmsetest_prev])

        print naccept / float(samples) * 100.0, '% was accepted'
        accept_ratio = naccept / (samples * 1.0) * 100

        # Close the files
        trainaccfile.close()
        testaccfile.close()
        trainrmsefile.close()
        testrmsefile.close()

        return (x_train, x_test, accept_ratio)

    def get_acc(self):
        self.train_acc = np.genfromtxt('./knowledge/trainacc.csv')
        self.test_acc = np.genfromtxt('./knowledge/testacc.csv')
        self.rmse_train = np.genfromtxt('./knowledge/trainrmse.csv')
        self.rmse_test = np.genfromtxt('./knowledge/testrmse.csv')

    def display_acc(self):
        burnin = 0.1 * self.samples  # use post burn in samples
        self.get_acc()
        rmse_tr = np.mean(self.rmse_train[int(burnin):])
        rmsetr_std = np.std(self.rmse_train[int(burnin):])

        rmse_tes = np.mean(self.rmse_test[int(burnin):])
        rmsetest_std = np.std(self.rmse_test[int(burnin):])

        print "Train accuracy:"

        print "Mean: " + str(np.mean(self.train_acc[int(burnin):]))
        print "\nTest accuracy:"
        print "Mean: " + str(np.mean(self.test_acc[int(burnin):]))

    def plot_acc(self, dataset):
        ax = plt.subplot(111)
        plt.plot(range(len(self.train_acc)), self.train_acc, 'g', label="train")
        plt.plot(range(len(self.test_acc)), self.test_acc, 'm', label="test")

        leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)

        plt.xlabel('Samples')
        plt.ylabel('Accuracy')
        plt.title(dataset + ' Accuracy plot')
        plt.savefig('./results/accuracy'+ dataset+'-mcmc.png')

        plt.clf()

        ax = plt.subplot(111)
        plt.plot(range(len(self.rmse_train)), self.rmse_train, 'b.', label="train-rmse")
        plt.plot(range(len(self.rmse_test)), self.rmse_test, 'c.', label="test-rmse")

        leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)

        plt.xlabel('Samples')
        plt.ylabel('RMSE')
        plt.title(dataset+' RMSE plot')
        plt.savefig('./results/rmse-'+dataset+'-mcmc.png')
        plt.clf()


def main():

    input = 11
    hidden = 94
    output = 10



    #if os.path.isfile("Results/"+filenames[problem]+"_rmse.txt"):
    #    print filenames[problem]
    #    continue

    traindata = np.genfromtxt('../datasets/WineQualityDataset/preprocess/winequality-white-train.csv', delimiter=',')
    testdata = np.genfromtxt('../datasets/WineQualityDataset/preprocess/winequality-white-test.csv', delimiter=',')


    topology = [input, hidden, output]

    random.seed(time.time())

    numSamples = 1000   # need to decide yourself

    mcmc = MCMC(numSamples, traindata, testdata, topology)  # declare class

    # generate random weights
    w_random = np.random.randn(mcmc.wsize)

    # start sampling
    mcmc.sampler(w_random, w_limit=0.05, transfer=True)

    burnin = 0.1 * numSamples  # use post burn in samples



if __name__ == "__main__": main()
