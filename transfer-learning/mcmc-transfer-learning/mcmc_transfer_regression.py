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
    def __init__(self, samples, sources, traindata, testdata, targettraindata, targettestdata, topology):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        self.targettraindata = targettraindata
        self.targettestdata = targettestdata
        self.numSources = sources
        self.wsize = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        self.createNetworks()
        # ----------------

    def createNetworks(self):
        self.sources = []
        for index in xrange(self.numSources):
            self.sources.append(Network(self.topology, self.traindata[index], self.testdata[index]))
        self.target = Network(self.topology, self.targettraindata, self.targettestdata)

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


    def sampler(self, w_pretrain, directory, save_knowledge=False):

        # Create file objects to write the attributes of the samples
        self.directory = directory
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        # trainrmsefile = open(self.directory+'/trainrmse.csv', 'w')
        # testrmsefile = open(self.directory+'/testrmse.csv', 'w')
        # targetrmsefile = open(self.directory+'/targetrmse.csv', 'w')



        # ------------------- initialize MCMC

        start = time.time()
        trainsize = np.zeros((self.numSources))
        testsize = np.zeros((self.numSources))
        for index in xrange(self.numSources):
            trainsize[index] = self.traindata[index].shape[0]
            testsize[index] = self.testdata[index].shape[0]

        targettrainsize = self.targettraindata.shape[0]
        targettestsize = self.targettestdata.shape[0]

        samples = self.samples

        # x_test = np.linspace(0, 1, num=testsize)
        # x_train = np.linspace(0, 1, num=trainsize)
        # x_target = np.linspace(0, 1, num=targetsize)

        y_train = []
        y_test = []
        netw = self.topology  # [input, hidden, output]
        for index in xrange(self.numSources):
            y_test.append(self.testdata[index][:, netw[0]:])
            y_train.append(self.traindata[index][:, netw[0]:])

        y_test_target = self.targettestdata[:, netw[0]:]
        y_train_target = self.targettraindata[:, netw[0]:]


        pos_w = np.ones((self.wsize, ))  # posterior of all weights and bias over all samples

        fxtrain_samples = []
        fxtest_samples = []
        for index in xrange(self.numSources):
            fxtrain_samples.append(np.ones((int(samples), int(trainsize[index]), netw[2])))  # fx of train data over all samples
            fxtest_samples.append(np.ones((int(samples), int(testsize[index]), netw[2])))  # fx of test data over all samples

        fxtarget_train_samples = np.ones((samples, targettrainsize, netw[2])) #fx of target train data over all samples
        fxtarget_test_samples = np.ones((samples, targettestsize, netw[2])) #fx of target test data over all samples

        w = np.zeros((self.numSources, self.wsize))
        w_proposal = np.zeros((self.numSources, self.wsize))
        for index in xrange(self.numSources):
            w[index] = w_pretrain
            w_proposal[index] = w_pretrain

        w_target = w_pretrain
        w_target_pro = w_pretrain

        step_w = 0.02  # defines how much variation you need in changes to w
        step_eta = 0.01
        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0

        # -------------------------------------initialize-------------------------
        pred_train = []
        pred_test = []
        eta = np.zeros((self.numSources))
        tau_pro = np.zeros((self.numSources))

        for index in xrange(self.numSources):
            pred_train.append(self.sources[index].evaluate_proposal(self.traindata[index], w[index]))
            pred_test.append(self.sources[index].evaluate_proposal(self.testdata[index], w[index]))
            eta[index] = np.log(np.var(pred_train[index] - y_train[index]))
            tau_pro[index] = np.exp(eta[index])


        pred_train_target = self.target.evaluate_proposal(self.targettraindata, w_target)
        pred_test_target = self.target.evaluate_proposal(self.targettestdata, w_target)

        eta_target = np.log(np.var(pred_train_target - y_train_target))
        tau_pro_target = np.exp(eta_target)

        prior = np.zeros((self.numSources))
        likelihood = np.zeros((self.numSources))
        likelihood_proposal = np.zeros((self.numSources))

        rmsetrain = np.zeros((self.numSources))
        rmsetest = np.zeros((self.numSources))

        for index in xrange(self.numSources):
            prior[index] = self.log_prior(sigma_squared, nu_1, nu_2, w[index], tau_pro[index])  # takes care of the gradients
            [likelihood[index], pred_train[index], rmsetrain[index]] = self.likelihood_func(self.sources[index], self.traindata[index], w[index], tau_pro[index])
            [likelihood_ignore, pred_test[index], rmsetest[index]] = self.likelihood_func(self.sources[index], self.testdata[index], w[index], tau_pro[index])

        [likelihood_target, pred_train_target, rmse_train_target] = self.likelihood_func(self.target, self.targettraindata, w_target, tau_pro_target)
        [likelihood_ignore, pred_test_target, rmse_test_target] = self.likelihood_func(self.target, self.targettestdata, w_target, tau_pro_target)



        # save the information
        # if save_knowledge:
        #     np.reshape(w_proposal, (1, w_proposal.shape[0]))
        #     with open(self.directory+'/wprop.csv', 'w') as wprofile:
        #         np.savetxt(wprofile, [w_proposal], delimiter=',', fmt='%.5f')
        #
        # np.savetxt(trainrmsefile, [rmsetrain])
        # np.savetxt(testrmsefile, [rmsetest])
        # np.savetxt(targetrmsefile, [rmsetarget])
        #
        # rmsetarget_prev = rmsetarget
        # rmsetest_prev = rmsetest
        # rmsetrain_prev = rmsetrain
        # wpro_prev = w_proposal

        naccept = 0
        # print 'begin sampling using mcmc random walk'

        prior_prop = np.zeros((self.numSources))

        for i in range(samples - 1):

            w_proposal = w + np.random.normal(0, step_w, self.wsize)
            w_target_pro = w_target + np.random.normal(0, step_w, self.wsize)

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            eta_pro_target = eta_target + np.random.normal(0, step_eta, 1)

            print eta_pro
            tau_pro = np.exp(eta_pro)
            tau_pro_target = np.exp(eta_pro_target)

            for index in xrange(self.numSources):
                [likelihood_proposal[index], pred_train[index], rmsetrain[index]] = self.likelihood_func(self.sources[index], self.traindata[index], w_proposal[index], tau_pro[index])
                [likelihood_ignore, pred_test[index], rmsetest[index]] = self.likelihood_func(self.sources[index], self.testdata[index], w_proposal[index], tau_pro[index])

            [likelihood_target_prop, pred_train_target, rmse_train_target] = self.likelihood_func(self.target, self.targettraindata, w_target_pro, tau_pro_target)
            [likelihood_ignore, pred_test_target, rmse_test_target] = self.likelihood_func(self.target, self.targettestdata, w_target_pro, tau_pro_target)


            # likelihood_ignore  refers to parameter that will not be used in the alg.
            for index in xrange(self.numSources):
                prior_prop[index] = self.log_prior(sigma_squared, nu_1, nu_2 ,w_proposal[index], tau_pro[index])  # takes care of the gradients


            diff_likelihood = likelihood_proposal - likelihood
            diff_prior = prior_prop - prior
            diff = np.zeros(diff_prior.shape)
            mh_prob = np.zeros(diff.shape)

            for index in xrange(self.numSources):
                diff[index] = min(700, diff_likelihood[index] + diff_prior[index])
                mh_prob[index] = min(1, math.exp(diff[index]))

            u = random.uniform(0, 1)

            for index in xrange(self.numSources):
                if u < mh_prob[index]:
                    # Update position
                    naccept += 1
                    likelihood[index] = likelihood_proposal[index]
                    prior[index] = prior_prop[index]
                    w[index] = w_proposal[index]
                    eta[index] = eta_pro[index]

                    # print i, trainacc, rmsetrain
                    elapsed = convert_time(time.time() - start)
                    sys.stdout.write(
                        '\rSamples: ' + str(i + 2) + "/" + str(samples)
                        + " Train RMSE: "+ str(rmsetrain)
                        + " Test RMSE: " + str(rmsetest)
                        + "\tTime elapsed: " + str(elapsed[0]) + ":" + str(elapsed[1]) )
    #                print ""

                    # save arrays to file
                    # if save_knowledge:
                    #     np.reshape(w_proposal, (1, w_proposal.shape[0]))
                    #     with open(self.directory+'/wprop.csv', 'w') as wprofile:
                    #             np.savetxt(wprofile, [w_proposal], delimiter=',', fmt='%.5f')

                    # print(pred_train[0], y_train[0])
                    fxtrain_samples[index][i + 1, :, :] = pred_train[index][:,  :]
                    fxtest_samples[index][i + 1, :, :] = pred_test[index][:, :]

                    # np.savetxt(trainrmsefile, [rmsetrain])
                    # np.savetxt(testrmsefile, [rmsetest])
                    # np.savetxt(targetrmsefile, [rmsetarget])

                    #save values into previous variables
                    # wpro_prev = w_proposal
                    # rmsetrain_prev = rmsetrain
                    # rmsetest_prev = rmsetest
                    # rmsetarget_prev = rmsetarget

                else:
                    # if save_knowledge:
                    #     np.reshape(wpro_prev, (1, wpro_prev.shape[0]))
                    #     with open(self.directory+'/wprop.csv', 'w') as wprofile:
                    #         np.savetxt(wprofile, [wpro_prev], delimiter=',', fmt='%.5f')
                    fxtrain_samples[index][i + 1, :, :] = fxtrain_samples[index][i, :, :]
                    fxtest_samples[index][i + 1, :, :] = fxtest_samples[index][i, :, :]
                    # np.savetxt(trainrmsefile, [rmsetrain_prev])
                    # np.savetxt(testrmsefile, [rmsetest_prev])
                    # np.savetxt(targetrmsefile, [rmsetarget_prev])


        print naccept / float(samples) * 100.0, '% was accepted'
        accept_ratio = naccept / (samples * 1.0) * 100

        # Close the files
        # trainrmsefile.close()
        # testrmsefile.close()
        # targetrmsefile.close()

        return (fxtrain_samples, fxtest_samples, accept_ratio)

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
        print "Test rmse:"
        print "Mean: " + str(rmse_tes) + " Std: " + str(rmsetest_std)
        print "Target rmse:"
        print "Mean: " + str(rmse_target) + " Std: " + str(rmsetarget_std)
        print "\n"

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

    numSources = 3
    building_id = [1]
    floor_id  = [0, 1, 2]
    traindata = []
    testdata = []

    for index in xrange(numSources):
        traindata.append(np.genfromtxt('../../datasets/UJIndoorLoc/trainingData/'+str(building_id[0])+str(floor_id[index])+'.csv',
                            delimiter=',')[:, :-2])
        testdata.append(np.genfromtxt('../../datasets/UJIndoorLoc/validationData/'+str(building_id[0])+str(floor_id[index])+'.csv',
                            delimiter=',')[:, :-2])

    targettraindata = np.genfromtxt('../../datasets/UJIndoorLoc/validationData/13.csv', delimiter=',')[:, :-2]
    targettestdata = np.genfromtxt('../../datasets/UJIndoorLoc/validationData/13.csv', delimiter=',')[:, :-2]


    # y_train = traindata[:, input:]
    # y_test = testdata[:, input:]
    # y_target = targetdata[:, input:]

    random.seed(time.time())

    numSamples = 2000# need to decide yourself
    burnin = 0.1 * numSamples

    mcmc_task = MCMC(numSamples, numSources, traindata, testdata, targettraindata, targettestdata, topology)  # declare class

    # generate random weights
    w_random = np.random.randn(mcmc_task.wsize)

    # start sampling
    x_train, x_test, fx_train, fx_test, accept_ratio = mcmc_task.sampler(w_random, save_knowledge=True, directory='target13')

    # display train and test accuracies
    mcmc_task.display_rmse()

    # Plot the accuracies and rmse
    mcmc_task.plot_rmse('Wifi Loc Task '+str(building_id)+str(floor))
