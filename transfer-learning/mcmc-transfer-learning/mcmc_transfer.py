# !/usr/bin/python
import matplotlib
matplotlib.use('Agg')
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

# --------------------------------------------- Basic Neural Network Class ---------------------------------------------

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

        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
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

    def BackwardPass(self, Input, desired):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)

    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))


        self.B1 = w[w_layer1size + w_layer2size :w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1] :w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]


    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        w = np.concatenate([w1, w2, self.B1, self.B2])
        return w


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
        self.B1 = self.BestB1
        self.B2 = self.BestB2  # load best knowledge

        for s in xrange(0, testSize):

            Input[:] = Data[s, 0:self.Top[0]]
            Desired[:] = Data[s, self.Top[0]:]

            self.ForwardPass(Input)
            sse = sse + self.sampleEr(Desired)

            if (np.isclose(self.out, Desired, atol=erTolerance).all()):
                clasPerf = clasPerf + 1

        return (sse / testSize, float(clasPerf) / testSize * 100)

# ------------------------------------------------------- MCMC Class --------------------------------------------------
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


    def sampler(self, w_pretrain, transfer = False):

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

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)



        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]:]
        y_train = self.traindata[:, netw[0]:]

        pos_w = np.ones((self.wsize, ))  # posterior of all weights and bias over all samples


        w = w_pretrain

        w_proposal = w_pretrain

        step_w = 0.02  # defines how much variation you need in changes to w

        # --------------------- Declare FNN and initialize

        neuralnet = Network(self.topology, self.traindata, self.testdata)

        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)


        sigma_squared = 25

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

            w_proposal = w + np.random.normal(0, step_w, self.wsize)

            [likelihood_proposal, pred_train, rmsetrain, trainacc] = self.likelihood_func(neuralnet, self.traindata, w_proposal)
            [likelihood_ignore, pred_test, rmsetest, testacc] = self.likelihood_func(neuralnet, self.testdata, w_proposal)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, w_proposal)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_prior = prior_prop - prior

            mh_prob = min(1, math.exp(diff_likelihood + diff_prior))

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


# ------------------------------------------------------- Main --------------------------------------------------------

if __name__ == '__main__':

    input = 11
    hidden = 94
    output = 10
    topology = [input, hidden, output]

    etol_tr = 0.2
    etol = 0.6
    alpha = 0.1

    MinCriteria = 0.005  # stop when RMSE reaches MinCriteria ( problem dependent)
    c = 1.2


    #--------------------------------------------- Train for the source task -------------------------------------------
    traindata = np.genfromtxt('../../datasets/WineQualityDataset/preprocess/winequality-white-train.csv', delimiter=',')
    testdata = np.genfromtxt('../../datasets/WineQualityDataset/preprocess/winequality-white-test.csv', delimiter=',')

    random.seed(time.time())

    numSamples = 20000# need to decide yourself

    mcmc_white = MCMC(numSamples, traindata, testdata, topology)  # declare class

    # # generate random weights
    # w_random = np.random.randn(mcmc.wsize)
    #
    # # start sampling
    # mcmc.sampler(w_random, transfer=True)
    #
    # # display train and test accuracies
    # mcmc.display_acc()
    #
    # # Plot the accuracies and rmse
    # mcmc.plot_acc('Wine-Quality-White')





    #------------------------------- Transfer weights from trained network to Target Task ---------------------------------
    w_transfer = mcmc_white.transfer(c)

    # Train for the target task with transfer
    traindata = np.genfromtxt('../../datasets/WineQualityDataset/preprocess/winequality-red-train.csv', delimiter=',')
    testdata = np.genfromtxt('../../datasets/WineQualityDataset/preprocess/winequality-red-test.csv', delimiter=',')

    random.seed(time.time())
    numSamples = 10000  # need to decide yourself

    # Create mcmc object for the target task
    mcmc_red_trf = MCMC(numSamples, traindata, testdata, topology)

    # start sampling
    mcmc_red_trf.sampler(w_transfer, transfer=False)

    # display train and test accuracies
    mcmc_red_trf.display_acc()

    # Plot the accuracies and rmse
    # mcmc.plot_acc('Wine-Quality-red')

    #------------------------------------------- Target Task Without Transfer-------------------------------------------

    random.seed(time.time())

    # Create mcmc object for the target task
    mcmc_red = MCMC(numSamples, traindata, testdata, topology)

    # generate random weights
    w_random = np.random.randn(mcmc_red.wsize)

    # start sampling
    mcmc_red.sampler(w_random, transfer=False)

    # display train and test accuracies
    mcmc_red.display_acc()


    # ----------------------------------------- Plot results of Transfer -----------------------------------------------

    ax = plt.subplot(111)
    plt.plot(range(len(mcmc_red.train_acc)), mcmc_red.train_acc, color='#FA7949', label="no-transfer")
    plt.plot(range(len(mcmc_red_trf.train_acc)), mcmc_red_trf.train_acc, color='#1A73B4', label="transfer")

    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.xlabel('Samples')
    plt.ylabel('Accuracy')
    plt.title('Wine Quality Red Train Accuracy plot')
    plt.savefig('./results/accuracy-train-mcmc.png')

    plt.clf()

    ax = plt.subplot(111)
    plt.plot(range(len(mcmc_red.test_acc)), mcmc_red.test_acc, color='#FA7949', label="no-transfer")
    plt.plot(range(len(mcmc_red_trf.test_acc)), mcmc_red_trf.test_acc, color='#1A73B4', label="transfer")

    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.xlabel('Samples')
    plt.ylabel('Accuracy')
    plt.title('Wine Quality Red Test Accuracy plot')
    plt.savefig('./results/accuracy-test-mcmc.png')

    plt.clf()

    ax = plt.subplot(111)
    plt.plot(range(len(mcmc_red.rmse_train)), mcmc_red.rmse_train, 'b', label="no-transfer")
    plt.plot(range(len(mcmc_red_trf.rmse_train)), mcmc_red_trf.rmse_train, 'c', label="transfer")

    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.xlabel('Samples')
    plt.ylabel('RMSE')
    plt.title('Wine Quality Red Train RMSE plot')
    plt.savefig('./results/rmse-train-mcmc.png')
    plt.clf()

    ax = plt.subplot(111)
    plt.plot(range(len(mcmc_red.rmse_test)), mcmc_red.rmse_test, 'b', label="no-transfer")
    plt.plot(range(len(mcmc_red_trf.rmse_test)), mcmc_red_trf.rmse_test, 'c', label="transfer")

    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.xlabel('Samples')
    plt.ylabel('RMSE')
    plt.title('Wine Quality Red Test RMSE plot')
    plt.savefig('./results/rmse-test-mcmc.png')
    plt.clf()

