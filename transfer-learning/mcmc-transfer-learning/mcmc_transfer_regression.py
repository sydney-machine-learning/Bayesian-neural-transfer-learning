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
    def __init__(self, samples, traindata, testdata, targetdata, topology):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        self.wsize = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        self.targetdata = targetdata
        # ----------------

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, data, w, tausq):
        y = data[:, self.topology[0]:]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [np.sum(loss), fx, rmse]

    def log_prior(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def genweights(self, w_mean, w_std):
        w_prop = np.ones(w_mean.shape)
        for index in range(w_mean.shape[0]):
            w_prop[index] = np.random.normal(w_mean[index], w_std[index], 1)
        return w_prop


    def transfer(self, stdmulconst):
        file = open(self.directory+'/wprop.csv', 'rb')
        lines = file.read().split('\n')[:-1]

        weights = np.ones((self.samples, self.wsize))
        burnin = int(0.1 * self.samples)

        for index in range(len(lines)):
            line = lines[index]
            w = np.array(list(map(float, line.split(','))))
            weights[index, :] = w

        weights = weights[burnin:, :]
        w_mean = weights.mean(axis=0)
        w_std = stdmulconst * np.std(weights, axis=0)
        return self.genweights(w_mean, w_std)


    def sampler(self, w_pretrain, directory, transfer=False):

        # Create file objects to write the attributes of the samples
        self.directory = directory
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        trainrmsefile = open(self.directory+'/trainrmse.csv', 'w')
        testrmsefile = open(self.directory+'/testrmse.csv', 'w')
        targetrmsefile = open(self.directory+'/targetrmse.csv', 'w')



        # ------------------- initialize MCMC

        start = time.time()
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        targetsize = self.targetdata.shape[0]
        samples = self.samples

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)
        x_target = np.linspace(0, 1, num=targetsize)

        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]:]
        y_train = self.traindata[:, netw[0]:]


        pos_w = np.ones((self.wsize, ))  # posterior of all weights and bias over all samples

        fxtrain_samples = np.ones((samples, trainsize, netw[2]))  # fx of train data over all samples
        fxtest_samples = np.ones((samples, testsize, netw[2]))  # fx of test data over all samples
        fxtarget_samples = np.ones((samples, targetsize, netw[2])) #fx of target data over all samples

        w = w_pretrain

        w_proposal = w_pretrain

        step_w = 0.02  # defines how much variation you need in changes to w
        step_eta = 0.01;

        # --------------------- Declare FNN and initialize

        neuralnet = Network(self.topology, self.traindata, self.testdata)

        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)
        pred_target = neuralnet.evaluate_proposal(self.targetdata, w)



        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0

        prior = self.log_prior(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)
        [likelihood_ignore, pred_target, rmsetarget] = self.likelihood_func(neuralnet, self.targetdata, w, tau_pro)



        # save the information
        if transfer:
            np.reshape(w_proposal, (1, w_proposal.shape[0]))
            with open(self.directory+'/wprop.csv', 'w') as wprofile:
                np.savetxt(wprofile, [w_proposal], delimiter=',', fmt='%.5f')

        np.savetxt(trainrmsefile, [rmsetrain])
        np.savetxt(testrmsefile, [rmsetest])
        np.savetxt(targetrmsefile, [rmsetarget])

        rmsetarget_prev = rmsetarget
        rmsetest_prev = rmsetest
        rmsetrain_prev = rmsetrain
        wpro_prev = w_proposal

        naccept = 0
        # print 'begin sampling using mcmc random walk'

        for i in range(samples - 1):

            w_proposal = w + np.random.normal(0, step_w, self.wsize)

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)


            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w_proposal, tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w_proposal, tau_pro)
            [likelihood_ignore, pred_trget, rmsetarget] = self.likelihood_func(neuralnet, self.targetdata, w_proposal, tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.log_prior(sigma_squared, nu_1, nu_2 ,w_proposal, tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_prior = prior_prop - prior

            diff = min(700, diff_likelihood + diff_prior)
            # print(diff)

            mh_prob = min(1, math.exp(diff))

            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                naccept += 1
                likelihood = likelihood_proposal
                prior = prior_prop
                w = w_proposal
                eta = eta_pro

                # print i, trainacc, rmsetrain
                elapsed = convert_time(time.time() - start)
                sys.stdout.write(
                    '\rSamples: ' + str(i + 2) + "/" + str(samples)
                    + " Train RMSE: "+ str(rmsetrain)
                    + " Test RMSE: " + str(rmsetest)
                    + "\tTime elapsed: " + str(elapsed[0]) + ":" + str(elapsed[1]) )
                print ""

                # save arrays to file
                if transfer:
                    np.reshape(w_proposal, (1, w_proposal.shape[0]))
                    with open(self.directory+'/wprop.csv', 'w') as wprofile:
                        np.savetxt(wprofile, [w_proposal], delimiter=',', fmt='%.5f')

                # print(pred_train[0], y_train[0])
                fxtrain_samples[i + 1, :, :] = pred_train[:,  :]
                fxtest_samples[i + 1, :, :] = pred_test[:, :]
                fxtarget_samples[i + 1, :, :] = pred_target[:, :]
                np.savetxt(trainrmsefile, [rmsetrain])
                np.savetxt(testrmsefile, [rmsetest])
                np.savetxt(targetrmsefile, [rmsetarget])

                #save values into previous variables
                wpro_prev = w_proposal
                rmsetrain_prev = rmsetrain
                rmsetest_prev = rmsetest
                rmsetarget_prev = rmsetarget

            else:
                if transfer:
                    np.reshape(wpro_prev, (1, wpro_prev.shape[0]))
                    with open(self.directory+'/wprop.csv', 'w') as wprofile:
                        np.savetxt(wprofile, [wpro_prev], delimiter=',', fmt='%.5f')
                fxtrain_samples[i + 1,:, :] = fxtrain_samples[i, :, :]
                fxtest_samples[i + 1,:, :] = fxtest_samples[i, :, :]
                fxtarget_samples[i + 1, :, :] = fxtarget_samples[i, :, :]
                np.savetxt(trainrmsefile, [rmsetrain_prev])
                np.savetxt(testrmsefile, [rmsetest_prev])
                np.savetxt(targetrmsefile, [rmsetarget_prev])


        print naccept / float(samples) * 100.0, '% was accepted'
        accept_ratio = naccept / (samples * 1.0) * 100

        # Close the files
        trainrmsefile.close()
        testrmsefile.close()
        targetrmsefile.close()

        return (x_train, x_test, fxtrain_samples, fxtest_samples, accept_ratio)

    def get_fx_rmse(self):
        self.rmse_train = np.genfromtxt(self.directory+'/trainrmse.csv')
        self.rmse_test = np.genfromtxt(self.directory+'/testrmse.csv')
        self.rmse_target = np.genfromtxt(self.directory+'/targetrmse.csv')

    def display_rmse(self):
        burnin = 0.1 * self.samples  # use post burn in samples
        self.get_fx_rmse()
        rmse_tr = np.mean(self.rmse_train[int(burnin):])
        rmsetr_std = np.std(self.rmse_train[int(burnin):])

        rmse_tes = np.mean(self.rmse_test[int(burnin):])
        rmsetest_std = np.std(self.rmse_test[int(burnin):])

        rmse_target = np.mean(self.rmse_target[int(burnin):])
        rmsetarget_std = np.std(self.rmse_target[int(burnin):])

        print "Train rmse:"
        print "Mean: " + str(rmse_tr) + " Std: " + str(rmsetr_std)
        print "\nTest rmse:"
        print "Mean: " + str(rmse_tes) + " Std: " + str(rmsetest_std)
        print "\nTarget rmse:"
        print "Mean: " + str(rmse_target) + " Std: " + str(rmsetarget_std)

    def plot_rmse(self, dataset):
        if not os.path.isdir(self.directory+'/results'):
            os.mkdir(self.directory+'/results')

        ax = plt.subplot(111)
        plt.plot(range(len(self.rmse_train)), self.rmse_train, 'b.', label="train-rmse")
        plt.plot(range(len(self.rmse_test)), self.rmse_test, 'c.', label="test-rmse")
        plt.plot(range(len(self.rmse_target)), self.rmse_target, 'm.', label="target-rmse")


        leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)

        plt.xlabel('Samples')
        plt.ylabel('RMSE')
        plt.title(dataset+' RMSE plot')
        plt.savefig(self.directory+'/results/rmse-'+dataset+'-mcmc.png')
        plt.clf()


# ------------------------------------------------------- Main --------------------------------------------------------

if __name__ == '__main__':

    input = 520
    hidden = 35
    output = 2
    topology = [input, hidden, output]

    MinCriteria = 0.005  # stop when RMSE reaches MinCriteria ( problem dependent)
    c = 1.2

    numTasks = 29

    #--------------------------------------------- Train for the source task -------------------------------------------

    for building_id in [0,1,2]:
        for floor in [1,2,3]:
            traindata = np.genfromtxt('../../datasets/UJIndoorLoc/trainingData/'+str(building_id)+str(floor)+'.csv', delimiter=',')
            testdata = np.genfromtxt('../../datasets/UJIndoorLoc/validationData/'+str(building_id)+str(floor)+'.csv', delimiter=',')
            targetdata = np.genfromtxt('../../datasets/UJIndoorLoc/validationData/00.csv', delimiter=',')

            traindata = traindata[:, :-2]
            testdata = testdata[:, :-2]
            targetdata = targetdata[:, :-2]

            y_train = traindata[:, input:]
            y_test = testdata[:, input:]
            y_target = targetdata[:, input:]


            random.seed(time.time())

            numSamples = 2000# need to decide yourself
            burnin = 0.1 * numSamples

            mcmc_task = MCMC(numSamples, traindata, testdata, targetdata, topology)  # declare class

            # generate random weights
            w_random = np.random.randn(mcmc_task.wsize)

            # start sampling
            x_train, x_test, fx_train, fx_test, accept_ratio = mcmc_task.sampler(w_random, transfer=True, directory='loc_'+str(building_id)+str(floor))

            # display train and test accuracies
            mcmc_task.display_rmse()

            # Plot the accuracies and rmse
            mcmc_task.plot_rmse('Wifi Loc Task '+str(building_id)+str(floor))


    # fx_train = fx_train[int(burnin):]
    # fx_test = fx_test[int(burnin):]
    #
    # fx_train_mu = fx_train.mean(axis=0)
    # fx_test_mu = fx_test.mean(axis=0)
    #
    # fx_high_tr = np.percentile(fx_train, 95, axis=0)
    # fx_low_tr = np.percentile(fx_train, 5, axis=0)
    #
    # fx_high = np.percentile(fx_test, 95, axis=0)
    # fx_low = np.percentile(fx_test, 5, axis=0)
    #
    #
    # ax = plt.subplot(111)
    # plt.plot(x_train, fx_train_mu[:, 0], 'b', label="fx_mu_train")
    # plt.plot(x_train, y_train[:, 0], 'c', label="y_train")
    # plt.plot(x_train, fx_high_tr[:, 0], 'g', label="fx_95_train")
    # plt.plot(x_train, fx_low_tr[:, 0], 'y', label="fx_5_train")
    #
    # leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    # leg.get_frame().set_alpha(0.5)
    #
    # plt.xlabel('Samples')
    # plt.ylabel('Longitude')
    # plt.title('Longitude plot')
    # plt.savefig(mcmc_task.directory+'/results/fx-longitude-train-mcmc.png')
    # plt.clf()
    #
    #
    # ax = plt.subplot(111)
    # plt.plot(x_train, fx_train_mu[:, 1], 'b', label="fx_mu_train")
    # plt.plot(x_train, y_train[:, 1], 'c', label="y_train")
    # plt.plot(x_train, fx_high_tr[:, 1], 'g', label="fx_95_train")
    # plt.plot(x_train, fx_low_tr[:, 1], 'y', label="fx_5_train")
    #
    # leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    # leg.get_frame().set_alpha(0.5)
    #
    # plt.xlabel('Samples')
    # plt.ylabel('Latitude')
    # plt.title('Latitude plot')
    # plt.savefig(mcmc_task.directory+'/results/fx-latitude-train-mcmc.png')
    # plt.clf()
    #
    # ax = plt.subplot(111)
    # plt.plot(x_test, fx_test_mu[:, 0], 'b', label="fx_mu_test")
    # plt.plot(x_test, y_test[:, 0], 'c', label="y_test")
    # plt.plot(x_test, fx_high[:, 0], 'g', label="fx_95_test")
    # plt.plot(x_test, fx_low[:, 0], 'y', label="fx_5_test")
    #
    # leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    # leg.get_frame().set_alpha(0.5)
    #
    # plt.xlabel('Samples')
    # plt.ylabel('Longitude')
    # plt.title('Longitude plot')
    # plt.savefig(mcmc_task.directory+'/results/fx-longitude-test-mcmc.png')
    # plt.clf()
    #
    #
    # ax = plt.subplot(111)
    # plt.plot(x_test, fx_test_mu[:, 1], 'b', label="fx_mu_test")
    # plt.plot(x_test, y_test[:, 1], 'c', label="y_test")
    # plt.plot(x_test, fx_high[:, 1], 'g', label="fx_95_test")
    # plt.plot(x_test, fx_low[:, 1], 'y', label="fx_5_test")
    #
    #
    # leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    # leg.get_frame().set_alpha(0.5)
    #
    # plt.xlabel('Samples')
    # plt.ylabel('Latitude')
    # plt.title('Latitude plot')
    # plt.savefig(mcmc_task.directory+'/results/fx-latitude-test-mcmc.png')
    # plt.clf()
