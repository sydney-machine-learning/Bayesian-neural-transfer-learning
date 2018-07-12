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
        x = x.astype(np.float128)
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
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        self.wsize = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        self.createNetworks()
        self.wsize_target = (self.targetTop[0] * self.targetTop[1]) + (self.targetTop[1] * self.targetTop[2]) + self.targetTop[1] + self.targetTop[2]

        # ----------------

    def report_progress(self, stdscr, sample_count, elapsed, rmsetrain, rmsetest, rmse_train_target, rmse_test_target, rmse_train_target_trf, rmse_test_target_trf, last_transfer, last_transfer_rmse, source_index, naccept_target_trf):
        stdscr.addstr(0, 0, "{} Samples Processed: {}/{} \tTime Elapsed: {}:{}".format(self.directory, sample_count, self.samples, elapsed[0], elapsed[1]))
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

    def createNetworks(self):
        self.sources = []
        for index in range(self.numSources):
            self.sources.append(Network(self.topology, self.traindata[index], self.testdata[index]))
        self.targetTop = self.topology.copy()
        self.targetTop[1] = int(1.0 * self.topology[1])
        self.target = Network(self.targetTop, self.targettraindata, self.targettestdata)


    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def nmse(self, predictions, targets):
        return np.sum((targets - predictions) ** 2)/np.sum((targets - np.mean(targets)) ** 2)

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


#     def transfer(self, w_transfer, w_pro_trf):
# #        # burnin = int(0.1 * self.samples)
# #        w_sum = np.zeros((self.wsize))
# #        std_sum = np.zeros((self.wsize))
#         indices = np.random.uniform(low=0, high=self.wsize, size=(self.transfersize)).astype('int')
#         w_pro = np.zeros(w_transfer.shape)
#         for index in range(w_transfer.shape[0]):
#             for transfer_index in indices:
#                 w_pro[index, :] = w_pro_trf
#                 w_pro[index, transfer_index] = w_transfer[index, transfer_index]
# #            weights = w_transfer[index, :]
# #            # w_mean = weights.mean(axis=0)
# #            # w_std = stdmulconst * np.std(weights, axis=0)
# #            w_sum += weights*trainsize[index]
# #            # std_sum += w_std*trainsize[index]
# #        w_mean = w_sum / float(np.sum(trainsize))
# #        # std_mean = w_sum / float(np.sum(trainsize))
# #        # return self.genweights(w_mean, std_mean)
#         return w_pro

    def transfer(self, w_sources, w_target):
        w_prop = np.zeros((w_sources.shape[0], w_target.shape[0]))
        for index in range(w_sources.shape[0]):
            w_prop[index, :] = w_target[:]
            # print(w_prop.shape)
            w_prop[index, :self.topology[0]*self.topology[1] + self.topology[1]] = w_sources[index, : self.topology[0]*self.topology[1] + self.topology[1]]
            w_prop[index, self.targetTop[0]*self.targetTop[1] + self.targetTop[1] : self.targetTop[0]*self.targetTop[1] + self.targetTop[1] + self.topology[1] * self.topology[2] + self.topology[2]] = w_sources[index, self.topology[0]*self.topology[1] +self.topology[1]:]
        return w_prop


    def mh_transfer(self, weights, eta, likelihood, prior, rmse_train, rmse_test):
        accept = False
        w_transfer = weights[0]
        rmse_tr = rmse_train
        rmse_tes = rmse_test
        eta_transfer = eta[0]
        for index in range(1, weights.shape[0]):
            tau = np.exp(eta[index])
            accept, rmsetrain, rmsetest, likelihood, prior = self.calc_mh_prob(self.target, self.targettraindata, self.targettestdata, weights[index], tau, likelihood, prior)

            if accept:
                w_transfer = weights[index]
                eta_transfer = eta[index]
                rmse_tr = rmsetrain
                rmse_tes = rmsetest

        return likelihood, prior, w_transfer, eta_transfer, rmse_tr, rmse_tes, accept

    def calc_mh_prob(self, network, traindata, testdata, weights, tau, likelihood, prior):
        accept = False
        [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(network, traindata, weights, tau)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(network, testdata, weights, tau)
        prior_prop = self.log_prior(self.sigma_squared, self.nu_1, self.nu_2, weights, tau)  # takes care of the gradients
        diff_likelihood = likelihood_proposal - likelihood
        diff_prior = prior_prop - prior
        diff = min(700, diff_likelihood + diff_prior)
        mh_prob = min(1, math.exp(diff))
        u = random.uniform(0, 1)
        if u < mh_prob:
            accept = True
            likelihood = likelihood_proposal
            prior = prior_prop

        return accept, rmsetrain, rmsetest, likelihood, prior


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



    def sampler(self, w_pretrain, w_pretrain_target, stdscr, save_knowledge=False, transfer='mh'):

        w_save = np.zeros(self.numSources + 2)
        weights_file = open('weights.csv', 'w')

        trainrmsefile = open(self.directory+'/trainrmse.csv', 'w')
        testrmsefile = open(self.directory+'/testrmse.csv', 'w')

        targettrainrmsefile = open(self.directory+'/targettrainrmse.csv', 'w')
        targettestrmsefile = open(self.directory+'/targettestrmse.csv', 'w')

        targettrftrainrmsefile = open(self.directory+'/targettrftrainrmse.csv', 'w')
        targettrftestrmsefile = open(self.directory+'/targettrftestrmse.csv', 'w')


        # initialize the number of samples
        # samples = self.samples


        # ------------------- initialize MCMC
        global start
        start = time.time()

        eta = np.zeros((self.numSources))
        tau_pro = np.zeros((self.numSources))
        prior = np.zeros((self.numSources))
        likelihood = np.zeros((self.numSources))
        likelihood_proposal = np.zeros((self.numSources))
        rmsetrain = np.zeros((self.numSources))
        rmsetest = np.zeros((self.numSources))
        rmsetrain_sample = np.zeros(rmsetrain.shape)
        rmsetest_sample = np.zeros(rmsetest.shape)
        trainsize = np.zeros((self.numSources))
        testsize = np.zeros((self.numSources))
        w = np.zeros((self.numSources, self.wsize))
        w_proposal = np.zeros((self.numSources, self.wsize))

        pred_train = []
        pred_test = []

        self.step_w = 0.02  # defines how much variation you need in changes to w
        self.step_eta = 0.01
        self.sigma_squared = 25
        self.nu_1 = 0
        self.nu_2 = 0

        fxtrain_samples = []
        fxtest_samples = []

        y_train = []
        y_test = []

        netw = self.topology  # [input, hidden, output]
        netw_target = self.targetTop

        for index in range(self.numSources):
            trainsize[index] = self.traindata[index].shape[0]
            testsize[index] = self.testdata[index].shape[0]
            y_test.append(self.testdata[index][:, netw[0]:])
            y_train.append(self.traindata[index][:, netw[0]:])
            # fxtrain_samples.append(np.ones((int(self.samples), int(trainsize[index]), netw[2])))  # fx of train data over all samples
            # fxtest_samples.append(np.ones((int(self.samples), int(testsize[index]), netw[2])))  # fx of test data over all samples
            w[index] = w_pretrain
            w_proposal[index] = w_pretrain
            pred_train.append(self.sources[index].evaluate_proposal(self.traindata[index], w[index]))
            pred_test.append(self.sources[index].evaluate_proposal(self.testdata[index], w[index]))
            eta[index] = np.log(np.var(pred_train[index] - y_train[index]))
            tau_pro[index] = np.exp(eta[index])
            prior[index] = self.log_prior(self.sigma_squared, self.nu_1, self.nu_2, w[index], tau_pro[index])  # takes care of the gradients
            [likelihood[index], pred_train[index], rmsetrain[index]] = self.likelihood_func(self.sources[index], self.traindata[index], w[index], tau_pro[index])
            [likelihood_ignore, pred_test[index], rmsetest[index]] = self.likelihood_func(self.sources[index], self.targettraindata, w[index], tau_pro[index])


        pos_w = np.ones((self.samples, self.numSources, self.wsize))  # posterior of all weights and bias over all samples
        # self.transfersize = netw[0] * netw[1]

        targettrainsize = self.targettraindata.shape[0]
        targettestsize = self.targettestdata.shape[0]
        y_test_target = self.targettestdata[:, netw[0]:]
        y_train_target = self.targettraindata[:, netw[0]:]
        # fxtarget_train_samples = np.ones((self.samples, targettrainsize, netw_target[2])) #fx of target train data over all samples
        # fxtarget_test_samples = np.ones((self.samples, targettestsize, netw_target[2])) #fx of target test data over all samples
        w_target = w_pretrain_target
        w_target_pro = w_pretrain_target
        pred_train_target = self.target.evaluate_proposal(self.targettraindata, w_target)
        pred_test_target = self.target.evaluate_proposal(self.targettestdata, w_target)
        eta_target = np.log(np.var(pred_train_target - y_train_target))
        tau_pro_target = np.exp(eta_target)
        prior_target = self.log_prior(self.sigma_squared, self.nu_1, self.nu_2, w_target, tau_pro_target)
        [likelihood_target, pred_train_target, rmse_train_target] = self.likelihood_func(self.target, self.targettraindata, w_target, tau_pro_target)
        [likelihood_ignore, pred_test_target, rmse_test_target] = self.likelihood_func(self.target, self.targettestdata, w_target, tau_pro_target)

        w_target_trf = w_target
        likelihood_target_trf = likelihood_target
        pred_train_target_trf = pred_train_target
        pred_test_target_trf = pred_test_target
        rmse_train_target_trf = rmse_train_target
        rmse_test_target_trf = rmse_test_target
        tau_pro_target_trf = tau_pro_target
        eta_target_trf = eta_target
        prior_target_trf = prior_target


        for index in range(self.numSources):
            rmsetrain_sample[index] = rmsetrain[index]
            rmsetest_sample[index] = rmsetest[index]
        rmsetrain_prev = rmsetrain
        rmsetest_prev = rmsetest

        # save the information
        np.savetxt(trainrmsefile, [rmsetrain_sample])
        np.savetxt(testrmsefile, [rmsetest_sample])

        np.savetxt(targettrainrmsefile, [rmse_train_target])
        np.savetxt(targettestrmsefile, [rmse_test_target])
        # save values into previous variables
        rmsetargettrain_prev = rmse_train_target
        rmsetargettest_prev = rmse_test_target

        np.savetxt(targettrftrainrmsefile, [rmse_train_target_trf])
        np.savetxt(targettrftestrmsefile, [rmse_test_target_trf])
        # save values into previous variables
        rmsetargettrftrain_prev = rmse_train_target_trf
        rmsetargettrftest_prev = rmse_test_target_trf

        naccept = np.zeros((self.numSources))
        naccept_target = 0
        naccept_target_trf = 0
        accept_target_trf = 0

        for index in range(self.numSources):
            w_save[index] = w[index, 0]
        w_save[self.numSources] = w_target[0]
        w_save[self.numSources+1] = w_target_trf[0]

        # print 'begin sampling using mcmc random walk'

        prior_prop = np.zeros((self.numSources))
        quantum = int( 0.01 * self.samples )

        last_transfer  = 0
        last_transfer_rmse = 0
        source_index = None

        for sample in range(self.samples - 1):

            w_proposal = w + np.random.normal(0, self.step_w, self.wsize)
            w_target_pro = w_target + np.random.normal(0, self.step_w, self.wsize_target)

            eta_pro = eta + np.random.normal(0, self.step_eta, 1)
            eta_pro_target = eta_target + np.random.normal(0, self.step_eta, 1)
            # print eta_pro
            tau_pro = np.exp(eta_pro)
            tau_pro_target = np.exp(eta_pro_target)

            if transfer != 'none':
                w_target_pro_trf = w_target_trf + np.random.normal(0, self.step_w, self.wsize_target)
                eta_pro_target_trf = eta_target_trf + np.random.normal(0, self.step_eta, 1)
                tau_pro_target_trf = np.exp(eta_pro_target_trf)


            # Check MH-acceptance probability for all source tasks
            for index in range(self.numSources):
                accept, rmsetrain[index], rmsetest[index], likelihood[index], prior[index] = self.calc_mh_prob(self.sources[index],
                                                            self.traindata[index], self.targettestdata, w_proposal[index], tau_pro[index],
                                                            likelihood[index], prior[index])
                if accept:
                    naccept[index] += 1
                    w[index] = w_proposal[index]
                    eta[index] = eta_pro[index]
                    rmsetrain_sample[index] = rmsetrain[index]
                    rmsetest_sample[index] = rmsetest[index]
                    rmsetrain_prev[index] = rmsetrain[index]
                    rmsetest_prev[index] = rmsetest[index]
                else:
                    rmsetrain_sample[index] = rmsetrain_prev[index]
                    rmsetest_sample[index] = rmsetest_prev[index]

            if save_knowledge:
                np.savetxt(trainrmsefile, [rmsetrain_sample])
                np.savetxt(testrmsefile, [rmsetest_sample])
                w_save[:self.numSources] = w[:, 0]

            # Check MH-acceptance probability for target task
            accept, rmse_train_target, rmse_test_target, likelihood_target, prior_target = self.calc_mh_prob(self.target,
                                                            self.targettraindata, self.targettestdata, w_target_pro, tau_pro_target,
                                                            likelihood_target, prior_target)
            if accept:
                naccept_target += 1
                w_target = w_target_pro
                eta_target = eta_pro_target
                rmsetargettrain_prev = rmse_train_target
                rmsetargettest_prev = rmse_test_target

            if save_knowledge:
                np.savetxt(targettrainrmsefile, [rmsetargettrain_prev])
                np.savetxt(targettestrmsefile, [rmsetargettest_prev])
                w_save[self.numSources] = w_target[0]


            # If transfer is True, evaluate proposal for target task with transfer
            if transfer != 'none':
                if sample != 0 and sample% quantum == 0 :
                    if transfer == 'mh':
                        w_sample = np.vstack([w_target_trf, w_target_pro_trf, w])
                        eta_sample = np.vstack([eta_target_trf, eta_pro_target_trf, eta])
                        likelihood_target_trf, prior_target_trf, w_target_trf, eta_target_trf, rmsetargettrftrain_prev, rmsetargettrftest_prev, accept = self.mh_transfer(w_sample.copy(),
                                                                                                                                                  eta_sample,
                                                                                                                                                  likelihood_target_trf,
                                                                                                                                                prior_target_trf, rmsetargettrftrain_prev, rmsetargettrftest_prev)
                    elif transfer == 'best':
                        w_sample = np.vstack([w, w_proposal])
                        best_w, best_rmse, transfer_index = self.find_best(w_sample, y_train_target)
                        w_sample = np.vstack([w_target_trf, w_target_pro, best_w])
                        likelihood_target_trf, prior_target_trf, w_target_trf, eta_target_trf, rmsetargettrftrain_prev, rmsetargettrftest_prev, accept = self.mh_transfer(w_sample.copy(),
                                                                                                                                                  tau_pro_target_trf,
                                                                                                                                                  likelihood_target_trf,
                                                                                                                                                prior_target_trf, rmsetargettrftrain_prev, rmsetargettrftest_prev)
                    if accept: accept_target_trf = sample


                else:
                    accept, rmse_train_target_trf, rmse_test_target_trf, likelihood_target_trf, prior_target_trf = self.calc_mh_prob(self.target, self.targettraindata, self.targettestdata, w_target_pro_trf, tau_pro_target_trf, likelihood_target_trf, prior_target_trf)

                    if accept:
                        naccept_target_trf += 1
                        w_target_trf = w_target_pro_trf
                        eta_target_trf = eta_pro_target_trf
                        # save values into previous variables
                        rmsetargettrftrain_prev = rmse_train_target_trf
                        rmsetargettrftest_prev = rmse_test_target_trf

                if save_knowledge:
                    np.savetxt(targettrftrainrmsefile, [rmsetargettrftrain_prev])
                    np.savetxt(targettrftestrmsefile, [rmsetargettrftest_prev])
                    w_save[self.numSources + 1] = w_target_trf[0]
                    np.savetxt(weights_file, [w_save], delimiter=',')

            elapsed = convert_time(time.time() - start)
            self.report_progress(stdscr, sample, elapsed, rmsetrain_sample, rmsetest_sample, rmsetargettrain_prev, rmsetargettest_prev, rmsetargettrftrain_prev, rmsetargettrftest_prev, last_transfer, last_transfer_rmse, source_index, accept_target_trf)

        elapsed = time.time() - start
        stdscr.clear()
        stdscr.refresh()
        stdscr.addstr(0 ,0 , r"Sampling Done!, {} % samples were accepted, Total Time: {}".format(np.array([naccept_target, naccept_target_trf]) / float(self.samples) * 100.0, elapsed))

        accept_ratio = naccept / (self.samples * 1.0) * 100

        # Close the files
        trainrmsefile.close()
        testrmsefile.close()
        targettrainrmsefile.close()
        targettestrmsefile.close()
        targettrftrainrmsefile.close()
        targettrftestrmsefile.close()
        weights_file.close()

        return (accept_ratio)



    def get_rmse(self):
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


    def display_rmse(self):
        burnin = 0.1 * self.samples  # use post burn in samples
        self.get_rmse()

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

        stdscr.addstr(2, 0, "Train rmse:")
        stdscr.addstr(3, 4, "Mean: " + str(rmse_tr) + " Std: " + str(rmsetr_std))
        stdscr.addstr(4, 0, "Test rmse:")
        stdscr.addstr(5, 4, "Mean: " + str(rmse_tes) + " Std: " + str(rmsetest_std))
        stdscr.addstr(7, 0, "Target Train rmse w/o transfer:")
        stdscr.addstr(8, 4, "Mean: " + str(rmse_target_train) + " Std: " + str(rmsetarget_std_train))
        stdscr.addstr(10, 0, "Target Test rmse w/o transfer:")
        stdscr.addstr(11, 4, "Mean: " + str(rmse_target_test) + " Std: " + str(rmsetarget_std_test))
        stdscr.addstr(13, 0, "Target Train rmse w/ transfer:")
        stdscr.addstr(14, 4, "Mean: " + str(rmse_target_train_trf) + " Std: " + str(rmsetarget_std_train_trf))
        stdscr.addstr(16, 0, "Target Test rmse w/ transfer:")
        stdscr.addstr(17, 4, "Mean: " + str(rmse_target_test_trf) + " Std: " + str(rmsetarget_std_test_trf))
        stdscr.getkey()
        stdscr.refresh()


    def plot_rmse(self, dataset):
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
# ------------------------------------------------------- Main --------------------------------------------------------

if __name__ == '__main__':

    input = 520
    hidden = 45
    output = 2
    topology = [input, hidden, output]

    MinCriteria = 0.005  # stop when RMSE reaches MinCriteria ( problem dependent)
    c = 1.2

    numTasks = 29
    start = None
    #--------------------------------------------- Train for the source task -------------------------------------------

    numSources = 1
#    building_id = [0, 1, 2]
#    floor_id  = [0, 1, 2, 3]

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()

    try:
        for index in [2]:
            targettraindata = np.genfromtxt('../../datasets/UJIndoorLoc/targetData/'+str(index)+'train.csv', delimiter=',')[:, :-2]
            targettestdata = np.genfromtxt('../../datasets/UJIndoorLoc/targetData/'+str(index)+'test.csv', delimiter=',')[:, :-2]
            # targettraindata = np.genfromtxt('../../datasets/synthetic_data/target_train.csv', delimiter=',')
            # targettestdata = np.genfromtxt('../../datasets/synthetic_data/target_test.csv', delimiter=',')

            traindata = []
            testdata = []
            for i in range(numSources):
                traindata.append(np.genfromtxt('../../datasets/UJIndoorLoc/sourceData/'+str(index)+'train.csv', delimiter=',')[:, :-2])
                testdata.append(np.genfromtxt('../../datasets/UJIndoorLoc/targetData/'+str(index)+'test.csv', delimiter=',')[:, :-2])
                # traindata.append(np.genfromtxt('../../datasets/synthetic_data/source'+str(i+1)+'.csv', delimiter=','))
                # testdata.append(np.genfromtxt('../../datasets/synthetic_data/target_test.csv', delimiter=','))


            stdscr.clear()
            random.seed(time.time())

            numSamples = 4000# need to decide yourself


            mcmc_task = TransferLearningMCMC(numSamples, numSources, traindata, testdata, targettraindata, targettestdata, topology,  directory='test_'+str(index+1))  # declare class

            # generate random weights
            w_random = np.random.randn(mcmc_task.wsize)
            w_random_target = np.random.randn(mcmc_task.wsize_target)

            # start sampling
            accept_ratio = mcmc_task.sampler(w_random, w_random_target, save_knowledge=True, stdscr=stdscr, transfer='mh')
            # display train and test accuracies
            mcmc_task.display_rmse()


            # Plot the accuracies and rmse
            mcmc_task.plot_rmse('Wifi Loc '+str(index+1))

    finally:
        curses.echo()
        curses.nocbreak()
        curses.endwin()
