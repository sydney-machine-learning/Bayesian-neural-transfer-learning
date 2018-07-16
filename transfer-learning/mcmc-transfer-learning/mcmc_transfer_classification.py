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
import curses


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

        for i in range(0, size):  # to see what fx is produced by your current weight update
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

        for s in range(0, testSize):

            Input[:] = Data[s, 0:self.Top[0]]
            Desired[:] = Data[s, self.Top[0]:]

            self.ForwardPass(Input)
            sse = sse + self.sampleEr(Desired)

            if (np.isclose(self.out, Desired, atol=erTolerance).all()):
                clasPerf = clasPerf + 1

        return (sse / testSize, float(clasPerf) / testSize * 100)

# ------------------------------------------------------- MCMC Class --------------------------------------------------
class TransferLearningMCMC:
    def __init__(self, samples, sources, traindata, testdata, targettraindata, targettestdata, topology, directory):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        self.targettraindata = targettraindata
        self.targettestdata = targettestdata
        self.numSources = sources

        # Create file objects to write the attributes of the samples
        self.directory = directory
        self.wsize = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        self.createNetworks()
        self.wsize_target = (self.targetTop[0] * self.targetTop[1]) + (self.targetTop[1] * self.targetTop[2]) + self.targetTop[1] + self.targetTop[2]
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

# ----------------

    def createNetworks(self):
        self.sources = []
        for index in range(self.numSources):
            self.sources.append(Network(self.topology, self.traindata[index], self.testdata[index]))
        self.targetTop = self.topology.copy()
        self.targetTop[1] = int(1.0 * self.topology[1])
        self.target = Network(self.targetTop, self.targettraindata, self.targettestdata)


    def report_progress(self, stdscr, sample_count, elapsed, rmsetrain, rmsetest, rmse_train_target, rmse_test_target, rmse_train_target_trf, rmse_test_target_trf, last_transfer, last_transfer_rmse, source_index, naccept_target_trf):
        stdscr.addstr(0, 0, "Samples Processed: {}/{} \tTime Elapsed: {}:{}".format(sample_count, self.samples, elapsed[0], elapsed[1]))
        i = 2
        index = 0
        for index in range(0, self.numSources):
            stdscr.addstr(index + i, 3, "Source {0} Progress:".format(index + 1))
            stdscr.addstr(index + i + 1, 5, "Train rmse: {:.4f}  Test rmse: {:.4f}".format(rmsetrain[index], rmsetest[index]))
            i += 2

        i = index + i + 2
        stdscr.addstr(i, 3, "Target w/o transfer Progress:")
        stdscr.addstr(i + 1, 5, "Train rmse: {:.4f}  Test rmse: {:.4f}".format(rmse_train_target, rmse_test_target))

        i += 4
        stdscr.addstr(i, 3, "Target w/ transfer Progress:")
        stdscr.addstr(i + 1, 5, "Train rmse: {:.4f}  Test rmse: {:.4f}".format(rmse_train_target_trf, rmse_test_target_trf))
        stdscr.addstr(i + 2, 5, "Last transfered sample: {} Last transfered rmse: {:.4f} Source index: {} last accept: {} ".format(last_transfer, last_transfer_rmse, source_index, naccept_target_trf) )

        stdscr.refresh()

    def softmax(self, fx):
        ex = np.exp(fx)
        sum_ex = np.sum(ex, axis = 1)
        sum_ex = np.multiply(np.ones(ex.shape), sum_ex[:, np.newaxis])
        prob = np.divide(ex, sum_ex)
        return prob

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def nmse(self, predictions, targets):
        return np.sum((targets - predictions) ** 2)/np.sum((targets - np.mean(targets)) ** 2)

    def likelihood_func(self, neuralnet, data, w):
        y = data[:, self.topology[0]:]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.nmse(fx, y)
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
        return [loss, rmse, acc]

    def find_best(self, weights, y):
        best_rmse = 999.9
        for index in range(weights.shape[0]):
            fx = self.target.evaluate_proposal(self.targettraindata, weights[index])
            rmse = self.rmse(fx, y)
            if rmse < best_rmse:
                best_rmse = rmse
                best_w = weights[index]
                best_index = index
        return best_w, best_rmse, best_index + 1


    def log_prior(self, sigma_squared, w):
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
        file = open(self.directory+'/wprop.csv', 'rb')
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

    def mh_transfer(self, weights, likelihood, prior, rmse_train, rmse_test, acc_train, acc_test):
        sigma_squared = 25
        accept = False
        w_transfer = weights[0]
        rmse_tr = rmse_train
        rmse_tes = rmse_test
        acc_tr = acc_train
        acc_tes = acc_test
        trf_index = -1
        for index in range(1, weights.shape[0]):
            [likelihood_proposal, rmsetrain, acctrain] = self.likelihood_func(self.target, self.targettraindata, weights[index])
            [likelihood_ignore, rmsetest, acctest] = self.likelihood_func(self.target, self.targettraindata, weights[index])
            prior_prop = self.log_prior(sigma_squared, weights[index])

            diff_likelihood = likelihood_proposal - likelihood
            diff_prior = prior_prop - prior
            diff = min(700, diff_likelihood + diff_prior)
            mh_prob = min(1, math.exp(diff))

            u = np.random.uniform(0,1)

            if u < mh_prob:
                likelihood = likelihood_proposal
                prior = prior_prop
                w_transfer = weights[index]
                rmse_tr = rmsetrain
                rmse_tes = rmsetest
                acc_tr = acctrain
                acc_tes = acctest
                accept = True
                trf_index = index

        return likelihood, prior, w_transfer, rmse_tr, rmse_tes, acc_tr, acc_tes, accept, trf_index



    def sampler(self, w_pretrain, w_pretrain_target, stdscr, save_knowledge=False):

        # Create file objects to write the attributes of the samples
        trainrmsefile = open(self.directory+'/trainrmse.csv', 'w')
        testrmsefile = open(self.directory+'/testrmse.csv', 'w')

        targettrainrmsefile = open(self.directory+'/targettrainrmse.csv', 'w')
        targettestrmsefile = open(self.directory+'/targettestrmse.csv', 'w')

        targettrftrainrmsefile = open(self.directory+'/targettrftrainrmse.csv', 'w')
        targettrftestrmsefile = open(self.directory+'/targettrftestrmse.csv', 'w')

        trainaccfile = open(self.directory+'/trainacc.csv', 'w')
        testaccfile = open(self.directory+'/testacc.csv', 'w')

        targettrainaccfile = open(self.directory+'/targettrainacc.csv', 'w')
        targettestaccfile = open(self.directory+'/targettestacc.csv', 'w')

        targettrftrainaccfile = open(self.directory+'/targettrftrainacc.csv', 'w')
        targettrftestaccfile = open(self.directory+'/targettrftestacc.csv', 'w')


        # ------------------- initialize MCMC

        start = time.time()
        trainsize = np.zeros((self.numSources))
        testsize = np.zeros((self.numSources))
        for index in range(self.numSources):
            trainsize[index] = self.traindata[index].shape[0]
            testsize[index] = self.testdata[index].shape[0]

        targettrainsize = self.targettraindata.shape[0]
        targettestsize = self.targettestdata.shape[0]

        samples = self.samples

        y_train = []
        y_test = []
        netw = self.topology  # [input, hidden, output]
        netw_target = self.targetTop
        for index in range(self.numSources):
            y_test.append(self.testdata[index][:, netw[0]:])
            y_train.append(self.traindata[index][:, netw[0]:])

        y_test_target = self.targettestdata[:, netw[0]:]
        y_train_target = self.targettraindata[:, netw[0]:]

        pos_w = np.ones((self.samples, self.numSources, self.wsize))  # posterior of all weights and bias over all samples
        # self.transfersize = netw[0] * netw[1]

        w = w_pretrain

        w_proposal = w_pretrain

        step_w = 0.02  # defines how much variation you need in changes to w

        # --------------------- Declare FNN and initialize
        w = np.zeros((self.numSources, self.wsize))
        w_proposal = np.zeros((self.numSources, self.wsize))
        for index in range(self.numSources):
            w[index] = w_pretrain
            w_proposal[index] = w_pretrain

        w_target = w_pretrain_target
        w_target_pro = w_pretrain_target

        step_w = 0.02  # defines how much variation you need in changes to w
        step_eta = 0.01
        sigma_squared = 25

        pred_train = []
        pred_test = []

        for index in range(self.numSources):
            pred_train.append(self.sources[index].evaluate_proposal(self.traindata[index], w[index]))
            pred_test.append(self.sources[index].evaluate_proposal(self.testdata[index], w[index]))
            # eta[index] = np.log(np.var(pred_train[index] - y_train[index]))
            # tau_pro[index] = np.exp(eta[index])


        pred_train_target = self.target.evaluate_proposal(self.targettraindata, w_target)
        pred_test_target = self.target.evaluate_proposal(self.targettestdata, w_target)

        # eta_target = np.log(np.var(pred_train_target - y_train_target))
        # tau_pro_target = np.exp(eta_target)

        prior = np.zeros((self.numSources))
        likelihood = np.zeros((self.numSources))
        likelihood_proposal = np.zeros((self.numSources))

        rmsetrain = np.zeros((self.numSources))
        rmsetest = np.zeros((self.numSources))
        acctrain = np.zeros((self.numSources))
        acctest = np.zeros((self.numSources))


        for index in range(self.numSources):
            prior[index] = self.log_prior(sigma_squared, w[index])  # takes care of the gradients
            [likelihood[index], rmsetrain[index], acctrain[index]] = self.likelihood_func(self.sources[index], self.traindata[index], w[index])
            [likelihood_ignore, rmsetest[index], acctest[index]] = self.likelihood_func(self.sources[index], self.targettraindata, w[index])

        prior_target = self.log_prior(sigma_squared, w_target)
        [likelihood_target, rmse_train_target, acc_train_target] = self.likelihood_func(self.target, self.targettraindata, w_target)
        [likelihood_ignore, rmse_test_target, acc_test_target] = self.likelihood_func(self.target, self.targettestdata, w_target)

        rmsetrain_sample = np.zeros(rmsetrain.shape)
        rmsetest_sample = np.zeros(rmsetest.shape)
        acctrain_sample = np.zeros(acctrain.shape)
        acctest_sample = np.zeros(acctest.shape)



        w_target_trf = w_target

        likelihood_target_trf = likelihood_target
        pred_train_target_trf = pred_train_target
        pred_test_target_trf = pred_test_target
        rmse_train_target_trf = rmse_train_target
        rmse_test_target_trf = rmse_test_target
        acc_train_target_trf = acc_train_target
        acc_test_target_trf = acc_test_target

        prior_target_trf = prior_target


        for index in range(self.numSources):
            rmsetrain_sample[index] = rmsetrain[index]
            rmsetest_sample[index] = rmsetest[index]
            acctrain_sample[index] = acctrain[index]
            acctest_sample[index] = acctest[index]



        rmsetrain_prev = rmsetrain
        rmsetest_prev = rmsetest
        acctrain_prev = acctrain
        acctest_prev = acctest

        # save the information
        np.savetxt(trainrmsefile, [rmsetrain_sample])
        np.savetxt(testrmsefile, [rmsetest_sample])
        np.savetxt(trainaccfile, [acctrain_sample])
        np.savetxt(testaccfile, [acctest_sample])

        np.savetxt(targettrainrmsefile, [rmse_train_target])
        np.savetxt(targettestrmsefile, [rmse_test_target])
        np.savetxt(targettrainaccfile, [acc_train_target])
        np.savetxt(targettestaccfile, [acc_test_target])

        # save values into previous variables
        rmsetargettrain_prev = rmse_train_target
        rmsetargettest_prev = rmse_test_target
        acctargettrain_prev = acc_train_target
        acctargettest_prev = acc_test_target


        np.savetxt(targettrftrainrmsefile, [rmse_train_target_trf])
        np.savetxt(targettrftestrmsefile, [rmse_test_target_trf])
        np.savetxt(targettrftrainaccfile, [acc_train_target_trf])
        np.savetxt(targettrftestaccfile, [acc_test_target_trf])

        # save values into previous variables
        rmsetargettrftrain_prev = rmse_train_target_trf
        rmsetargettrftest_prev = rmse_test_target_trf
        acctargettrftrain_prev = acc_train_target_trf
        acctargettrftest_prev = acc_test_target_trf

        naccept = np.zeros((self.numSources))
        naccept_target = 0
        naccept_target_trf = 0
        accept_target_trf = -1

        mh_prob = np.zeros(self.numSources)
        # print 'begin sampling using mcmc random walk'

        prior_prop = np.zeros((self.numSources))
        quantum = int( 0.01 * samples )

        last_transfer  = 0
        last_transfer_rmse = 0
        source_index = None
        # print 'begin sampling using mcmc random walk'

        for i in range(samples - 1):

            w_proposal = w + np.random.normal(0, step_w, self.wsize)
            w_target_pro = w_target + np.random.normal(0, step_w, self.wsize_target)
            w_target_pro_trf = w_target_trf + np.random.normal(0, step_w, self.wsize_target)

            for index in range(self.numSources):
                [likelihood_proposal[index], rmsetrain[index], acctrain[index]] = self.likelihood_func(self.sources[index], self.traindata[index], w_proposal[index])
                [likelihood_ignore, rmsetest[index], acctest[index]] = self.likelihood_func(self.sources[index], self.targettraindata, w_proposal[index])

            # likelihood_ignore  refers to parameter that will not be used in the alg.
            for index in range(self.numSources):
                prior_prop[index] = self.log_prior(sigma_squared, w_proposal[index])  # takes care of the gradients

            [likelihood_target_prop, rmse_train_target, acc_train_target] = self.likelihood_func(self.target, self.targettraindata, w_target_pro)
            [likelihood_ignore, rmse_test_target, acc_test_target] = self.likelihood_func(self.target, self.targettestdata, w_target_pro)

            prior_target_prop = self.log_prior(sigma_squared, w_target_pro)

            diff_likelihood_target = likelihood_target_prop - likelihood_target
            diff_prior_target = prior_target_prop - prior_target
            diff_target = min(700, diff_likelihood_target + diff_prior_target)
            mh_prob_target = min(1, math.exp(diff_target))


            diff_likelihood = likelihood_proposal - likelihood
            diff_prior = prior_prop - prior
            diff = np.zeros(diff_prior.shape)
            mh_prob = np.zeros(diff.shape)

            for index in range(self.numSources):
                diff[index] = min(700, diff_likelihood[index] + diff_prior[index])
                mh_prob[index] = min(1, math.exp(diff[index]))

            u = random.uniform(0, 1)
            for index in range(self.numSources):
                if u < mh_prob[index]:
                    # Update position
                    naccept[index] += 1
                    likelihood[index] = likelihood_proposal[index]
                    prior[index] = prior_prop[index]
                    w[index] = w_proposal[index]

                    # print i, trainacc, rmsetrain
                    elapsed = convert_time(time.time() - start)

                    rmsetrain_sample[index] = rmsetrain[index]
                    rmsetest_sample[index] = rmsetest[index]
                    acctrain_sample[index] = acctest[index]
                    acctest_sample[index] = acctest[index]

                    rmsetrain_prev[index] = rmsetrain[index]
                    rmsetest_prev[index] = rmsetest[index]
                    acctrain_prev[index] = acctrain[index]
                    acctest_prev[index] = acctest[index]


                else:
                    rmsetrain_sample[index] = rmsetrain_prev[index]
                    rmsetest_sample[index] = rmsetest_prev[index]
                    acctrain_sample[index] = acctrain_prev[index]
                    acctest_sample[index] = acctest_prev[index]

            np.savetxt(trainrmsefile, [rmsetrain_sample])
            np.savetxt(testrmsefile, [rmsetest_sample])
            np.savetxt(trainaccfile, [acctrain_sample])
            np.savetxt(testaccfile, [acctest_sample])

            u = random.uniform(0,1)
            if u < mh_prob_target:
                naccept_target += 1
                likelihood_target = likelihood_target_prop
                prior_target = prior_target_prop
                w_target = w_target_pro

                elapsed = convert_time(time.time() - start)
                if save_knowledge:
                    np.savetxt(targettrainrmsefile, [rmse_train_target])
                    np.savetxt(targettestrmsefile, [rmse_test_target])
                    np.savetxt(targettrainaccfile, [acc_train_target])
                    np.savetxt(targettestaccfile, [acc_test_target])

                    # save values into previous variables
                    rmsetargettrain_prev = rmse_train_target
                    rmsetargettest_prev = rmse_test_target
                    acctargettrain_prev = acc_train_target
                    acctargettest_prev = acc_test_target


            else:
                if save_knowledge:
                    np.savetxt(targettrainrmsefile, [rmsetargettrain_prev])
                    np.savetxt(targettestrmsefile, [rmsetargettest_prev])
                    np.savetxt(targettrainaccfile, [acctargettrain_prev])
                    np.savetxt(targettestaccfile, [acctargettest_prev])

            w_prop = w_target_pro_trf.copy()

            if i != 0 and i % quantum == 0:
                w_sample = np.vstack([w_target_trf, w_target_pro_trf, w_proposal])
                likelihood_target_trf, prior_target_trf, w_target_trf, rmsetargettrftrain_prev, rmsetargettrftest_prev, acctargettrftrain_prev, acctargettrftest_prev, accept, transfer_index = self.mh_transfer(w_sample.copy(),
                                                                                                                                                  likelihood_target_trf,
                                                                                                                                                  prior_target_trf,
                                                                                                                                                  rmsetargettrftrain_prev,
                                                                                                                                                  rmsetargettrftest_prev,
                                                                                                                                                  acctargettrftrain_prev, acctargettrftest_prev)
                last_transfer = i
                if accept:
                    accept_target_trf = i
                    last_transfer_rmse = rmsetargettrftrain_prev
                    source_index = transfer_index

                if save_knowledge:
                    np.savetxt(targettrftrainrmsefile, [rmsetargettrftrain_prev])
                    np.savetxt(targettrftestrmsefile, [rmsetargettrftest_prev])
                    np.savetxt(targettrftrainaccfile, [acctargettrftrain_prev])
                    np.savetxt(targettrftestaccfile, [acctargettrftest_prev])
            else:
                [likelihood_target_prop_trf, rmse_train_target_trf, acc_train_target_trf] = self.likelihood_func(self.target, self.targettraindata, w_prop)
                [likelihood_ignore_trf, rmse_test_target_trf, acc_test_target_trf] = self.likelihood_func(self.target, self.targettestdata, w_prop)

                prior_target_prop_trf = self.log_prior(sigma_squared, w_prop)

                diff_likelihood_target_trf = likelihood_target_prop_trf - likelihood_target_trf
                diff_prior_target_trf = prior_target_prop_trf - prior_target_trf
                diff_target_trf = min(700, diff_likelihood_target_trf + diff_prior_target_trf)
                mh_prob_target_trf = min(1, math.exp(diff_target_trf))


                u = random.uniform(0,1)
                if u < mh_prob_target_trf:
                    naccept_target_trf += 1
                    likelihood_target_trf = likelihood_target_prop_trf
                    prior_target_trf = prior_target_prop_trf
                    w_target_trf = w_prop
                    try:
                        if not np.array_equal(w_prop, w_target_pro_trf):
                            pass
                    except:
                        pass

                    if save_knowledge:
                        np.savetxt(targettrftrainrmsefile, [rmse_train_target_trf])
                        np.savetxt(targettrftestrmsefile, [rmse_test_target_trf])
                        np.savetxt(targettrftrainaccfile, [acc_train_target_trf])
                        np.savetxt(targettrftestaccfile, [acc_test_target_trf])

                        # save values into previous variables
                        rmsetargettrftrain_prev = rmse_train_target_trf
                        rmsetargettrftest_prev = rmse_test_target_trf
                        acctargettrftrain_prev = acc_train_target_trf
                        acctargettrftest_prev = acc_test_target_trf

                else:
                    if save_knowledge:
                        np.savetxt(targettrftrainrmsefile, [rmsetargettrftrain_prev])
                        np.savetxt(targettrftestrmsefile, [rmsetargettrftest_prev])
                        np.savetxt(targettrftrainaccfile, [acctargettrftrain_prev])
                        np.savetxt(targettrftestaccfile, [acctargettrftest_prev])

            elapsed = convert_time(time.time() - start)
            self.report_progress(stdscr, i, elapsed, rmsetrain_sample, rmsetest_sample, rmsetargettrain_prev, rmsetargettest_prev, rmsetargettrftrain_prev, rmsetargettrftest_prev, last_transfer, last_transfer_rmse, source_index, accept_target_trf)
        elapsed = time.time() - start
        stdscr.clear()
        stdscr.refresh()
        stdscr.addstr(0 ,0 , r"Sampling Done!, {} % samples were accepted, Time elapsed: {}".format(np.array([naccept_target, naccept_target_trf]) / float(samples) * 100.0, elapsed))

        accept_ratio = naccept / (samples * 1.0) * 100

        # Close the files
        trainrmsefile.close()
        testrmsefile.close()
        targettrainrmsefile.close()
        targettestrmsefile.close()
        targettrftrainrmsefile.close()
        targettrftestrmsefile.close()

        return (accept_ratio)


    def get_rmse_acc(self):
        self.rmse_train = np.genfromtxt(self.directory+'/trainrmse.csv')
        self.rmse_test = np.genfromtxt(self.directory+'/testrmse.csv')
        if self.numSources == 1:
            self.rmse_test = self.rmse_test.reshape((self.rmse_test.shape[0], 1))
            self.rmse_train = self.rmse_train.reshape((self.rmse_train.shape[0], 1))
        self.rmse_target_train = np.genfromtxt(self.directory+'/targettrainrmse.csv')
        self.rmse_target_test = np.genfromtxt(self.directory+'/targettestrmse.csv')
        self.rmse_target_train_trf = np.genfromtxt(self.directory+'/targettrftrainrmse.csv')
        self.rmse_target_test_trf = np.genfromtxt(self.directory+'/targettrftestrmse.csv')
        # print self.rmse_test.shape
        self.acc_train = np.genfromtxt(self.directory+'/trainacc.csv')
        self.acc_test = np.genfromtxt(self.directory+'/testacc.csv')
        if self.numSources == 1:
            self.acc_test = self.rmse_test.reshape((self.acc_test.shape[0], 1))
            self.acc_train = self.rmse_train.reshape((self.acc_train.shape[0], 1))
        self.acc_target_train = np.genfromtxt(self.directory+'/targettrainacc.csv')
        self.acc_target_test = np.genfromtxt(self.directory+'/targettestacc.csv')
        self.acc_target_train_trf = np.genfromtxt(self.directory+'/targettrftrainacc.csv')
        self.acc_target_test_trf = np.genfromtxt(self.directory+'/targettrftestacc.csv')


    def display_rmse_acc(self):
        burnin = 0.1 * self.samples  # use post burn in samples
        self.get_rmse_acc()

        rmse_tr = [0 for index in range(self.numSources)]
        rmsetr_std = [0 for index in range(self.numSources)]
        rmse_tes = [0 for index in range(self.numSources)]
        rmsetest_std = [0 for index in range(self.numSources)]

        for index in range(numSources):
            rmse_tr[index] = np.mean(self.rmse_train[int(burnin):, index])
            rmsetr_std[index] = np.std(self.rmse_train[int(burnin):, index])

            rmse_tes[index] = np.mean(self.rmse_test[int(burnin):, index])
            rmsetest_std[index] = np.std(self.rmse_test[int(burnin):, index])

        rmse_target_train = np.mean(self.rmse_target_train[int(burnin):])
        rmsetarget_std_train = np.std(self.rmse_target_train[int(burnin):])

        rmse_target_test = np.mean(self.rmse_target_test[int(burnin):])
        rmsetarget_std_test = np.std(self.rmse_target_test[int(burnin):])


        rmse_target_train_trf = np.mean(self.rmse_target_train_trf[int(burnin):])
        rmsetarget_std_train_trf = np.std(self.rmse_target_train_trf[int(burnin):])

        rmse_target_test_trf = np.mean(self.rmse_target_test_trf[int(burnin):])
        rmsetarget_std_test_trf = np.std(self.rmse_target_test_trf[int(burnin):])

        acc_tr = [0 for index in range(self.numSources)]
        acctr_std = [0 for index in range(self.numSources)]
        acc_tes = [0 for index in range(self.numSources)]
        acctest_std = [0 for index in range(self.numSources)]

        for index in range(numSources):
            acc_tr[index] = np.mean(self.acc_train[int(burnin):, index])
            acctr_std[index] = np.std(self.acc_train[int(burnin):, index])

            acc_tes[index] = np.mean(self.acc_test[int(burnin):, index])
            acctest_std[index] = np.std(self.acc_test[int(burnin):, index])

        acc_target_train = np.mean(self.acc_target_train[int(burnin):])
        acctarget_std_train = np.std(self.acc_target_train[int(burnin):])

        acc_target_test = np.mean(self.acc_target_test[int(burnin):])
        acctarget_std_test = np.std(self.acc_target_test[int(burnin):])


        acc_target_train_trf = np.mean(self.acc_target_train_trf[int(burnin):])
        acctarget_std_train_trf = np.std(self.acc_target_train_trf[int(burnin):])

        acc_target_test_trf = np.mean(self.acc_target_test_trf[int(burnin):])
        acctarget_std_test_trf = np.std(self.acc_target_test_trf[int(burnin):])

        # stdscr.addstr(2, 0, "Train rmse:")
        # stdscr.addstr(3, 4, "Mean: " + str(rmse_tr) + " Std: " + str(rmsetr_std))
        # stdscr.addstr(4, 0, "Test rmse:")
        # stdscr.addstr(5, 4, "Mean: " + str(rmse_tes) + " Std: " + str(rmsetest_std))
        stdscr.addstr(2, 0, "Target Train results w/o transfer:")
        stdscr.addstr(3, 4, "RMSE: " + str(rmse_target_train) + " Acc: " + str(acc_target_train))
        stdscr.addstr(4, 0, "Target Test results w/o transfer:")
        stdscr.addstr(5, 4, "RMSE: " + str(rmse_target_test) + " Acc: " + str(acc_target_test))
        stdscr.addstr(7, 0, "Target Train results w/ transfer:")
        stdscr.addstr(8, 4, "RMSE: " + str(rmse_target_train_trf) + " Acc: " + str(acc_target_train_trf))
        stdscr.addstr(9, 0, "Target Test results w/ transfer:")
        stdscr.addstr(10, 4, "RMSE: " + str(rmse_target_test_trf) + " Acc: " + str(acc_target_test_trf))
        stdscr.getkey()
        stdscr.refresh()


    def plot_rmse_acc(self, dataset):
        if not os.path.isdir(self.directory+'/results'):
            os.mkdir(self.directory+'/results')

        burnin = int(0.1 * self.samples)

        ax = plt.subplot(111)
        x = np.array(np.arange(burnin, self.samples))
        plt.plot(x, self.rmse_target_train[burnin: ], '.' , label="no-transfer")
        plt.plot(x, self.rmse_target_train_trf[burnin: ], '.' , label="transfer")
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('RMSE')
        plt.title(dataset+' Train RMSE')
        plt.savefig(self.directory+'/results/rmse-'+dataset+'train-mcmc.png')
        plt.clf()


        ax = plt.subplot(111)
        plt.plot(x, self.rmse_target_test[burnin: ], '.' , label="no-transfer")
        plt.plot(x, self.rmse_target_test_trf[burnin: ], '.' , label="transfer")
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('RMSE')
        plt.title(dataset+' Test RMSE')
        plt.savefig(self.directory+'/results/rmse-'+dataset+'test-mcmc.png')
        plt.clf()

        ax = plt.subplot(111)
        x = np.array(np.arange(burnin, self.samples))
        plt.plot(x, self.acc_target_train[burnin: ], '.' , label="no-transfer")
        plt.plot(x, self.acc_target_train_trf[burnin: ], '.' , label="transfer")
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('Accuracy')
        plt.title(dataset+' Train Accuracy')
        plt.savefig(self.directory+'/results/acc-'+dataset+'train-mcmc.png')
        plt.clf()


        ax = plt.subplot(111)
        plt.plot(x, self.acc_target_test[burnin: ], '.' , label="no-transfer")
        plt.plot(x, self.acc_target_test_trf[burnin: ], '.' , label="transfer")
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('Accuracy')
        plt.title(dataset+' Test Accuracy')
        plt.savefig(self.directory+'/results/acc-'+dataset+'test-mcmc.png')
        plt.clf()


# ------------------------------------------------------- Main --------------------------------------------------------

if __name__ == '__main__':

    input = 11
    hidden = 75
    output = 10
    topology = [input, hidden, output]

    MinCriteria = 0.005  # stop when RMSE reaches MinCriteria ( problem dependent)
    c = 1.2

    numTasks = 29

    #--------------------------------------------- Train for the source task -------------------------------------------

    numSources = 1
#    building_id = [0, 1, 2]
#    floor_id  = [0, 1, 2, 3]

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()

    try:

        targettraindata = np.genfromtxt('../../datasets/WineQualityDataset/preprocess/winequality-red-train.csv', delimiter=',')
        targettestdata = np.genfromtxt('../../datasets/WineQualityDataset/preprocess/winequality-red-test.csv', delimiter=',')
        traindata = []
        testdata = []
        # for index in range(numSources):
        traindata.append(np.genfromtxt('../../datasets/WineQualityDataset/preprocess/winequality-white-train.csv',
                                        delimiter=','))
        testdata.append(np.genfromtxt('../../datasets/WineQualityDataset/preprocess/winequality-red-test.csv',
                                        delimiter=','))

        stdscr.clear()
        random.seed(time.time())

        numSamples = 15000# need to decide yourself


        mcmc_task = TransferLearningMCMC(numSamples, numSources, traindata, testdata, targettraindata, targettestdata, topology,  directory='wine-quality-red')  # declare class

        # generate random weights
        w_random = np.random.randn(mcmc_task.wsize)
        w_random_target = np.random.randn(mcmc_task.wsize_target)

        # start sampling
        accept_ratio = mcmc_task.sampler(w_random, w_random_target, save_knowledge=True, stdscr=stdscr)
        # display train and test accuracies
        mcmc_task.display_rmse_acc()


        # Plot the accuracies and rmse
        mcmc_task.plot_rmse_acc('wine quality red')

    finally:
        curses.echo()
        curses.nocbreak()
        curses.endwin()
