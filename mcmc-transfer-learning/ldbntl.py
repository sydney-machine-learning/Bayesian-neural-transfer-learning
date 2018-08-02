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

class Network(object):

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

    @staticmethod
    def sigmoid(x):
        x = x.astype(np.float128)
        sig =  1 / (1 + np.exp(-x))
        return sig
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
        layer = 1  # hidden to output
        for x in range(0, self.Top[layer]):
            for y in range(0, self.Top[layer + 1]):
                self.W2[x, y] += self.lrate * out_delta[y] * self.hidout[x]
        for y in range(0, self.Top[layer + 1]):
            self.B2[y] += -1 * self.lrate * out_delta[y]

        layer = 0  # Input to Hidden
        for x in range(0, self.Top[layer]):
            for y in range(0, self.Top[layer + 1]):
                self.W1[x, y] += self.lrate * hid_delta[y] * Input[x]
        for y in range(0, self.Top[layer + 1]):
            self.B1[y] += -1 * self.lrate * hid_delta[y]

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

    @staticmethod
    def scaler(data, maxout=1, minout=0, maxin=1, minin=0):
        attribute = data[:]
        attribute = minout + (attribute - minin)*((maxout - minout)/(maxin - minin))
        return attribute

    @staticmethod
    def denormalize(data, indices, maxval, minval):
        for i in range(len(indices)):
            index = indices[i]
            attribute = data[:, index]
            attribute = Network.scaler(attribute, maxout=maxval[i], minout=minval[i], maxin=1, minin=0)
            data[:, index] = attribute
        return data

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

    def langevin_gradient(self, data, w, depth):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))

        batch_size = min(10, size)
        indices = np.random.randint(0, size, batch_size)
        # print(indices)

        for index in range(0, batch_size):
            pat = indices[index]
            Input = data[pat, 0:self.Top[0]]
            Desired = data[pat, self.Top[0]:self.Top[0]+self.Top[2]]
            self.ForwardPass(Input)
            self.BackwardPass(Input, Desired)

        w_updated = self.encode()

        return  w_updated


    @staticmethod
    def softmax(fx):
        ex = np.exp(fx)
        sum_ex = np.sum(ex, axis = 1)
        sum_ex = np.multiply(np.ones(ex.shape), sum_ex[:, np.newaxis])
        prob = np.divide(ex, sum_ex)
        return prob

# ------------------------------------------------------- MCMC Class --------------------------------------------------
class TransferLearningMCMC(object):
    def __init__(self, samples, sources, traindata, testdata, targettraindata, targettestdata, topology, directory, type='regression'):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        self.targettraindata = targettraindata
        self.targettestdata = targettestdata
        self.numSources = sources
        self.type = type
        self.sgd_depth = 1
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

    @staticmethod
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    @staticmethod
    def nmse(predictions, targets):
        return np.sum((targets - predictions) ** 2)/np.sum((targets - np.mean(targets)) ** 2)

    @staticmethod
    def distance(fx, y):
        dist = np.sqrt(np.sum(np.square(fx - y), axis=1)).mean()
        return dist

    @staticmethod
    def log_multivariate_normal(x, mean, sigma=0.02, size=100):
        diagmat_size = min(size, x.shape[0])
        sigma_diagmat = np.zeros((diagmat_size, diagmat_size))
        np.fill_diagonal(sigma_diagmat, sigma)
        indices = np.random.randint(0, x.shape[0], diagmat_size)
        x_mean = np.zeros(diagmat_size)
        x_val = np.zeros(diagmat_size)
        for i in range(diagmat_size):
            index = indices[i]
            x_mean[i] = mean[index].copy()
            x_val[i]= x[index].copy()
        return multivariate_normal.logpdf(x_val, mean=x_mean, cov=sigma_diagmat)


    def likelihood_func(self, neuralnet, data, w, tau):
        if self.type == 'regression':
            likelihood, rmse = self.gauss_likelihood(neuralnet, data, w, tau)
        elif self.type == 'classification':
            likelihood, rmse, acc = self.multi_likelihood(neuralnet, data, w)
        return likelihood, rmse

    def log_prior(self, w, tau):
        if self.type == 'regression':
            loss = self.reg_prior(self.sigma_squared, self.nu_1, self.nu_2, w, tau)
        elif self.type == 'classification':
            loss = self.class_prior(self.sigma_squared, w)
        return loss

    def multi_likelihood(self, neuralnet, data, w):
        y = data[:, self.topology[0]:]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.nmse(fx, y)
        prob = neuralnet.softmax(fx)
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

    def class_prior(self, sigma_squared, w):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2
        return log_loss

    def gauss_likelihood(self, neuralnet, data, w, tausq):
        y = data[:, neuralnet.Top[0]: neuralnet.Top[0] + neuralnet.Top[2]].copy()
        # y_m = data[:, 522:524]
        fx = neuralnet.evaluate_proposal(data, w)
        # fx_m = Network.denormalize(fx.copy(), [0,1], maxval=[-7299.786516730871000, 4865017.3646842018], minval=[-7695.9387549299299000, 4864745.7450159714])
        # y_m = Network.denormalize(y.copy(), [0,1], maxval=[-7299.786516730871000, 4865017.3646842018], minval=[-7695.9387549299299000, 4864745.7450159714])
        # np.savetxt('y.txt', y, delimiter=',')
        # rmse = self.distance(fx_m, y_m)
        rmse = TransferLearningMCMC.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [np.sum(loss), rmse]

    def reg_prior(self, sigma_squared, nu_1, nu_2, w, tausq):
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

    def transfer(self, weights, eta, likelihood, prior, rmse_train, rmse_test):
        accept = False
        w_transfer = weights[0]
        w_transfer_gd = weights[1]
        rmse_tr = rmse_train
        rmse_tes = rmse_test
        eta_transfer = eta[0]
        accept = False

        index =  np.random.uniform(2, self.numSources, 1).astype('int')
        source_proposal = weights[index][0]
        source_proposal_gd = weights[self.numSources + index][0]

        tau = np.exp(eta[self.numSources + index])
        sampleaccept, rmsetrain, rmsetest, likelihood, prior = self.transfer_prob(self.target, self.targettraindata, self.targettestdata, w_transfer, w_transfer_gd, source_proposal, source_proposal_gd, tau, likelihood, prior)

        if sampleaccept:
            w_transfer = source_proposal
            eta_transfer = eta[self.numSources + index]
            rmse_tr = rmsetrain
            rmse_tes = rmsetest
            accept = True

        return likelihood, prior, w_transfer, eta_transfer, rmse_tr, rmse_tes, accept, index


    def transfer_prob(self, network, traindata, testdata, w_current, w_current_gd, w_prop, w_prop_gd, tau, likelihood, prior):
        accept = False
        [likelihood_proposal, rmsetrain] = self.likelihood_func(network, traindata, w_prop, tau)
        [likelihood_ignore, rmsetest] = self.likelihood_func(network, testdata, w_prop, tau)
        prior_prop = self.log_prior(w_prop, tau)
        diff_likelihood = likelihood_proposal - likelihood
        diff_prior = prior_prop - prior

        trans_dist_prop = self.log_multivariate_normal(x=w_current, mean=w_prop_gd, sigma=self.sigma)
        trans_dist_curr = self.log_multivariate_normal(x=w_prop, mean=w_current_gd, sigma=self.sigma)

        transfer_diff = trans_dist_prop - trans_dist_curr
        diff = diff_likelihood + diff_prior + transfer_diff
        diff = diff.astype('float128')
        mh_transfer_prob = min(1, np.exp(diff))

        u = random.uniform(0, 1)
        if u < mh_transfer_prob:
            accept = True
            likelihood = likelihood_proposal
            prior = prior_prop

        return accept, rmsetrain, rmsetest, likelihood, prior


    def accept_prob(self, network, traindata, testdata, weights, weights_gd, weights_pro, weights_pro_gd, tau, likelihood, prior):
        accept = False
        [likelihood_proposal, rmsetrain] = self.likelihood_func(network, traindata, weights_pro, tau)
        [likelihood_ignore, rmsetest] = self.likelihood_func(network, testdata, weights_pro, tau)
        prior_prop = self.log_prior(weights_pro, tau)  # takes care of the gradients

        _q = self.log_multivariate_normal(x=weights, mean=weights_pro_gd, sigma=self.sigma)
        q = self.log_multivariate_normal(x=weights_pro, mean=weights_gd, sigma=self.sigma)

        diff_q = _q - q
        diff_likelihood = likelihood_proposal - likelihood
        diff_prior = prior_prop - prior
        diff = diff_likelihood + diff_prior + diff_q
        diff = diff.astype('float128')
        mh_prob = min(1, np.exp(diff))
        u = random.uniform(0, 1)
        if u < mh_prob:
            accept = True
            likelihood = likelihood_proposal
            prior = prior_prop

        return accept, rmsetrain, rmsetest, likelihood, prior



    def sampler(self, w_pretrain, w_pretrain_target, stdscr, save_knowledge=False, transfer='mh', quantum_coeff=0.01):

        w_save = np.zeros(self.numSources + 2)
        weights_file = open('weights.csv', 'w')

        trainrmsefile = open(self.directory+'/trainrmse.csv', 'w')
        testrmsefile = open(self.directory+'/testrmse.csv', 'w')

        targettrainrmsefile = open(self.directory+'/targettrainrmse.csv', 'w')
        targettestrmsefile = open(self.directory+'/targettestrmse.csv', 'w')

        targettrftrainrmsefile = open(self.directory+'/targettrftrainrmse.csv', 'w')
        targettrftestrmsefile = open(self.directory+'/targettrftestrmse.csv', 'w')

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
        self.sigma = 0.02

        y_train = []
        y_test = []

        netw = self.topology  # [input, hidden, output]
        netw_target = self.targetTop

        for index in range(self.numSources):
            trainsize[index] = self.traindata[index].shape[0]
            testsize[index] = self.testdata[index].shape[0]
            y_test.append(self.testdata[index][:, netw[0]:netw[0]+netw[2]])
            y_train.append(self.traindata[index][:, netw[0]:netw[0]+netw[2]])
            w[index] = w_pretrain
            w_proposal[index] = w_pretrain
            pred_train.append(self.sources[index].evaluate_proposal(self.traindata[index], w[index]))
            eta[index] = np.log(np.var(pred_train[index] - y_train[index]))
            tau_pro[index] = np.exp(eta[index])
            prior[index] = self.log_prior(w[index], tau_pro[index])  # takes care of the gradients
            [likelihood[index], rmsetrain[index]] = self.likelihood_func(self.sources[index], self.traindata[index], w[index], tau_pro[index])
            [likelihood_ignore, rmsetest[index]] = self.likelihood_func(self.sources[index], self.testdata[index], w[index], tau_pro[index])


        # pos_w = np.ones((self.samples, self.numSources, self.wsize))  # posterior of all weights and bias over all samples

        targettrainsize = self.targettraindata.shape[0]
        targettestsize = self.targettestdata.shape[0]
        y_test_target = self.targettestdata[:, netw[0]:netw[0]+netw[2]]
        y_train_target = self.targettraindata[:, netw[0]:netw[0]+netw[2]]
        w_target = w_pretrain_target
        w_target_pro = w_pretrain_target
        pred_train_target = self.target.evaluate_proposal(self.targettraindata, w_target)
        eta_target = np.log(np.var(pred_train_target - y_train_target))
        tau_pro_target = np.exp(eta_target)
        prior_target = self.log_prior(w_target, tau_pro_target)
        [likelihood_target, rmse_train_target] = self.likelihood_func(self.target, self.targettraindata, w_target, tau_pro_target)
        [likelihood_ignore, rmse_test_target] = self.likelihood_func(self.target, self.targettestdata, w_target, tau_pro_target)

        w_target_trf = w_target
        likelihood_target_trf = likelihood_target
        pred_train_target_trf = pred_train_target
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
        ntransfer = 0
        transfer_attempts = 0

        for index in range(self.numSources):
            w_save[index] = w[index, -1]
        w_save[self.numSources] = w_target[-1]
        w_save[self.numSources+1] = w_target_trf[-1]

        prior_prop = np.zeros((self.numSources))
        quantum = int( quantum_coeff * self.samples )

        last_transfer  = 0
        last_transfer_rmse = 0
        source_index = None
        w_gd = np.zeros((self.numSources, self.wsize))
        w_prop_gd = np.zeros((self.numSources, self.wsize))

        for sample in range(self.samples - 1):
            for index in range(self.numSources):
                w_gd[index] = self.sources[index].langevin_gradient(self.traindata[index], w[index].copy(), self.sgd_depth)
                w_proposal[index] = w_gd[index] + np.random.normal(0, self.step_w, self.wsize)
                w_prop_gd[index] = self.sources[index].langevin_gradient(self.traindata[index], w_proposal[index].copy(), self.sgd_depth)

            w_target_gd = self.target.langevin_gradient(self.targettraindata, w_target.copy(), self.sgd_depth)
            w_target_pro = w_target_gd + np.random.normal(0, self.step_w, self.wsize_target)
            w_target_pro_gd = self.target.langevin_gradient(self.targettraindata, w_target_pro.copy(), self.sgd_depth)

            eta_pro = eta + np.random.normal(0, self.step_eta, 1)
            eta_pro_target = eta_target + np.random.normal(0, self.step_eta, 1)

            tau_pro = np.exp(eta_pro)
            tau_pro_target = np.exp(eta_pro_target)

            w_target_trf_gd = self.target.langevin_gradient(self.targettraindata, w_target_trf.copy(), self.sgd_depth)
            w_target_pro_trf = w_target_trf_gd + np.random.normal(0, self.step_w, self.wsize_target)
            w_target_pro_trf_gd = self.target.langevin_gradient(self.targettraindata, w_target_pro_trf.copy(), self.sgd_depth)
            eta_pro_target_trf = eta_target_trf + np.random.normal(0, self.step_eta, 1)
            tau_pro_target_trf = np.exp(eta_pro_target_trf)


            # Check MH-acceptance probability for all source tasks
            for index in range(self.numSources):
                accept, rmsetrain[index], rmsetest[index], likelihood[index], prior[index] = self.accept_prob(self.sources[index],self.traindata[index], self.testdata[index], w[index], w_gd[index], w_proposal[index], w_prop_gd[index], tau_pro[index],likelihood[index], prior[index])
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
                w_save[:self.numSources] = w[:, -1]

            # Check MH-acceptance probability for target task
            accept, rmse_train_target, rmse_test_target, likelihood_target, prior_target = self.accept_prob(self.target, self.targettraindata, self.targettestdata, w_target, w_target_gd, w_target_pro, w_target_pro_gd, tau_pro_target, likelihood_target, prior_target)

            if accept:
                naccept_target += 1
                w_target = w_target_pro
                eta_target = eta_pro_target
                rmsetargettrain_prev = rmse_train_target
                rmsetargettest_prev = rmse_test_target

            if save_knowledge:
                np.savetxt(targettrainrmsefile, [rmsetargettrain_prev])
                np.savetxt(targettestrmsefile, [rmsetargettest_prev])
                w_save[self.numSources] = w_target[-1]


            # If transfer is True, evaluate proposal for target task with transfer
            if transfer != 'none':
                if sample != 0 and sample % quantum == 0:
                    accept = False
                    transfer_attempts += 1
                    last_transfer = sample
                    if transfer == 'mh':
                        w_sample = np.vstack([w_target_trf, w_target_trf_gd, w_proposal, w_prop_gd])
                        eta = eta.reshape((self.numSources,1))
                        eta_pro = eta_pro.reshape((self.numSources,1))
                        eta_sample = np.vstack([eta_target_trf, eta, eta_pro])
                        likelihood_target_trf, prior_target_trf, w_target_trf, eta_target_trf, rmsetargettrftrain_prev, rmsetargettrftest_prev, accept, transfer_index = self.transfer(w_sample.copy(), eta_sample,likelihood_target_trf, prior_target_trf, rmsetargettrftrain_prev, rmsetargettrftest_prev)

                    if accept:
                        accept_target_trf = sample
                        ntransfer += 1
                        last_transfer_rmse = rmsetargettrftrain_prev
                        source_index = transfer_index

                else:
                    accept, rmse_train_target_trf, rmse_test_target_trf, likelihood_target_trf, prior_target_trf = self.accept_prob(self.target, self.targettraindata, self.targettestdata, w_target_trf, w_target_trf_gd, w_target_pro_trf, w_target_pro_trf_gd, tau_pro_target_trf, likelihood_target_trf, prior_target_trf)

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
                    w_save[self.numSources + 1] = w_target_trf[-1]
                    np.savetxt(weights_file, [w_save], delimiter=',')

            elapsed = convert_time(time.time() - start)
            self.report_progress(stdscr, sample, elapsed, rmsetrain_sample, rmsetest_sample, rmsetargettrain_prev, rmsetargettest_prev, rmsetargettrftrain_prev, rmsetargettrftest_prev, last_transfer, last_transfer_rmse, source_index, accept_target_trf)

        accept_ratio_target = np.array([naccept_target, naccept_target_trf]) / float(self.samples) * 100
        elapsed = time.time() - start
        stdscr.clear()
        stdscr.refresh()
        stdscr.addstr(0 ,0 , r"Sampling Done!, {} % samples were accepted, Total Time: {}".format(accept_ratio_target, elapsed))

        accept_ratio = naccept / (self.samples * 1.0) * 100
        transfer_ratio = ntransfer / transfer_attempts * 100

        with open(self.directory+"/ratios.txt", 'w') as accept_ratios_file:
            for ratio in accept_ratio:
                accept_ratios_file.write(str(ratio) + ' ')
            for ratio in accept_ratio_target:
                accept_ratios_file.write(str(ratio) + ' ')
            accept_ratios_file.write(str(transfer_ratio) + ' ')


        # Close the files
        trainrmsefile.close()
        testrmsefile.close()
        targettrainrmsefile.close()
        targettestrmsefile.close()
        targettrftrainrmsefile.close()
        targettrftestrmsefile.close()
        weights_file.close()

        return (accept_ratio, transfer_ratio)



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

        for index in range(self.numSources):
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

        for index in range(self.numSources):
            ax = plt.subplot(111)
            x = np.array(np.arange(burnin, self.samples))
            plt.plot(x, self.rmse_train[burnin: , index], '.' , label="train")
            plt.plot(x, self.rmse_test[burnin: , index], '.' , label="test")
            plt.legend()
            plt.xlabel('Samples')
            plt.ylabel('RMSE')
            plt.title(dataset+' Source '+str(index+1)+' RMSE')
            plt.savefig(self.directory+'/results/rmse-source-'+str(index+1)+'.png')
            plt.clf()

        ax = plt.subplot(111)
        x = np.array(np.arange(burnin, self.samples))
        plt.plot(x, self.rmse_target_train[burnin: ], '.' , label="no-transfer")
        plt.plot(x, self.rmse_target_train_trf[burnin: ], '.' , label="transfer")
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('RMSE')
        plt.title(dataset+' Train RMSE')
        plt.savefig(self.directory+'/results/rmse-target-train-mcmc.png')
        plt.clf()


        ax = plt.subplot(111)
        plt.plot(x, self.rmse_target_test[burnin: ], '.' , label="no-transfer")
        plt.plot(x, self.rmse_target_test_trf[burnin: ], '.' , label="transfer")
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('RMSE')
        plt.title(dataset+' Test RMSE')
        plt.savefig(self.directory+'/results/rmse-target-test-mcmc.png')
        plt.clf()
# ------------------------------------------------------- Main --------------------------------------------------------

if __name__ == '__main__':

    name = ["Wine-Quality", "UJIndoorLoc", "Sarcos", "Synthetic"]
    input = [11, 520, 21, 4]
    hidden = [105, 140, 55, 25]
    output = [10, 2, 1, 1]
    numSources = [1, 1, 1, 5]
    type = {0:'classification', 1:'regression', 2:'regression', 3:'regression'}
    numSamples = [8000, 4000, 4000, 8000]

    problem = 3
    problemtype = type[problem]
    topology = [input[problem], hidden[problem], output[problem]]
    problem_name = name[problem]

    start = None

    #--------------------------------------------- Train for the source task -------------------------------------------

    stdscr = None
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()

    ntransferlist = []

    try:
        # stdscr.clear()
        # targettraindata = np.genfromtxt('../datasets/WineQualityDataset/preprocess/winequality-red-train.csv', delimiter=',')
        # targettestdata = np.genfromtxt('../datasets/WineQualityDataset/preprocess/winequality-red-test.csv', delimiter=',')
        # targettraindata = np.genfromtxt('../datasets/UJIndoorLoc/targetData/0train.csv', delimiter=',')[:, :-2]
        # targettestdata = np.genfromtxt('../datasets/UJIndoorLoc/targetData/0test.csv', delimiter=',')[:, :-2]
        targettraindata = np.genfromtxt('../datasets/synthetic_data/target_train.csv', delimiter=',')
        targettestdata = np.genfromtxt('../datasets/synthetic_data/target_test.csv', delimiter=',')
        # targettraindata = np.genfromtxt('../datasets/Sarcos/target_train.csv', delimiter=',')
        # targettestdata = np.genfromtxt('../datasets/Sarcos/target_test.csv', delimiter=',')

        traindata = []
        testdata = []
        for i in range(numSources[problem]):
            # traindata.append(np.genfromtxt('../datasets/WineQualityDataset/preprocess/winequality-white-train.csv', delimiter=','))
            # testdata.append(np.genfromtxt('../datasets/WineQualityDataset/preprocess/winequality-red-test.csv', delimiter=','))
            # traindata.append(np.genfromtxt('../datasets/UJIndoorLoc/sourceData/'+str(i)+'train.csv', delimiter=',')[:, :-2])
            # testdata.append(np.genfromtxt('../datasets/UJIndoorLoc/sourceData/'+str(i)+'test.csv', delimiter=',')[:, :-2])
            traindata.append(np.genfromtxt('../datasets/synthetic_data/source'+str(i+1)+'.csv', delimiter=','))
            testdata.append(np.genfromtxt('../datasets/synthetic_data/target_test.csv', delimiter=','))
            # traindata.append(np.genfromtxt('../datasets/Sarcos/source.csv', delimiter=','))
            # testdata.append(np.genfromtxt('../datasets/Sarcos/target_test.csv', delimiter=','))
            pass

        # stdscr.clear()
        random.seed(time.time())

        mcmc_task = TransferLearningMCMC(numSamples[problem], numSources[problem], traindata, testdata, targettraindata, targettestdata, topology,  directory=problem_name, type=problemtype)  # declare class

        # generate random weights
        w_random = np.random.randn(mcmc_task.wsize)
        w_random_target = np.random.randn(mcmc_task.wsize_target)

        # start sampling
        accept_ratio, transfer_ratio = mcmc_task.sampler(w_random, w_random_target, save_knowledge=True, stdscr=stdscr, transfer='mh', quantum_coeff=0.005)

        # display trai
            # testdata.append(np.genn and test accuracies
        mcmc_task.display_rmse()

        # Plot the accuracies and rmse
        mcmc_task.plot_rmse(problem_name)

    finally:
        curses.echo()
        curses.nocbreak()
        curses.endwin()
        pass
