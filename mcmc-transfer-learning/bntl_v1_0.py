# !/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import multivariate_normal
from scipy.stats import norm
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

    @staticmethod
    def softmax(fx):
        ex = np.exp(fx)
        sum_ex = np.sum(ex, axis = 1)
        sum_ex = np.multiply(np.ones(ex.shape), sum_ex[:, np.newaxis])
        prob = np.divide(ex, sum_ex)
        return prob

# ------------------------------------------------------- MCMC Class --------------------------------------------------
class BayesianTL(object):
    def __init__(self, num_samples, num_sources, train_data, test_data, target_train_data, target_test_data, topology, directory, type='regression'):
        self.num_samples = num_samples  # NN topology [input, hidden, output]
        self.source_topology = topology  # max epocs
        self.source_train_data = train_data  #
        self.source_test_data = test_data
        self.target_train_data = target_train_data
        self.target_test_data = target_test_data
        self.num_sources = num_sources
        self.type = type
        self.directory = directory
        # Create file objects to write the attributes of the samples
        self.source_wsize = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        self.create_networks()
        self.target_wsize = (self.target_topology[0] * self.target_topology[1]) + (self.target_topology[1] * self.target_topology[2]) + self.target_topology[1] + self.target_topology[2]

        # ----------------

    @staticmethod
    def create_directory(directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)


    def report_progress(self, stdscr, sample_count, elapsed, rmse_train_source, rmse_test_source, rmse_train_target, rmse_test_target, rmse_train_target_trf, rmse_test_target_trf, last_transfer_sample, last_transfer_rmse, source_index, naccept_target_trf):
        stdscr.addstr(0, 0, "{} Samples Processed: {}/{} \tTime Elapsed: {}:{}".format(self.directory, sample_count, self.num_samples, elapsed[0], elapsed[1]))
        i = 2
        index = 0
        for index in range(0, self.num_sources):
            stdscr.addstr(index + i, 3, "Source {0} Progress:".format(index + 1))
            stdscr.addstr(index + i + 1, 5, "Train rmse: {:.4f}  Test rmse: {:.4f}".format(rmse_train_source[index], rmse_test_source[index]))
            i += 2

        i = index + i + 2
        stdscr.addstr(i, 3, "Target w/o transfer Progress:")
        stdscr.addstr(i + 1, 5, "Train rmse: {:.4f}  Test rmse: {:.4f}".format(rmse_train_target, rmse_test_target))

        i += 4
        stdscr.addstr(i, 3, "Target w/ transfer Progress:")
        stdscr.addstr(i + 1, 5, "Train rmse: {:.4f}  Test rmse: {:.4f}".format(rmse_train_target_trf, rmse_test_target_trf))
        stdscr.addstr(i + 2, 5, "Last transfered sample: {} Last transfered rmse: {:.4f} Source index: {} last accept: {} ".format(last_transfer_sample, last_transfer_rmse, source_index, naccept_target_trf) )

        stdscr.refresh()

    def create_networks(self):
        self.sources = []
        for index in range(self.num_sources):
            self.sources.append(Network(self.source_topology, self.source_train_data[index], self.source_test_data[index]))
        self.target_topology = self.source_topology.copy()
        self.target_topology[1] = int(1.0 * self.source_topology[1])
        self.target = Network(self.target_topology, self.target_train_data, self.target_test_data)

    @staticmethod
    def calculate_rmse(predictions, desired):
        return np.sqrt(((predictions - desired) ** 2).mean())

    @staticmethod
    def calculate_nmse(predictions, desired):
        return np.sum((desired - predictions) ** 2)/np.sum((desired - np.mean(desired)) ** 2)

    # Calculates Euclidian distance between points
    @staticmethod
    def calculate_distance(prediction, desired):
        distance = np.sqrt(np.sum(np.square(prediction - desired), axis=1)).mean()
        return distance


    def likelihood_function(self, neural_network, data, weights, tau):
        if self.type == 'regression':
            likelihood, rmse = self.gaussian_likelihood(neural_network, data, weights, tau)
        elif self.type == 'classification':
            likelihood, rmse, accuracy = self.multinomial_likelihood(neural_network, data, weights)
        return likelihood, rmse

    def prior_function(self, weights, tau):
        if self.type == 'regression':
            loss = self.gaussian_prior(self.sigma_squared, self.nu_1, self.nu_2, weights, tau)
        elif self.type == 'classification':
            loss = self.classification_prior(self.sigma_squared, weights)
        return loss

    @staticmethod
    def multinomial_likelihood(neural_network, data, weights):
        y = data[:, neural_network.Top[0]: neural_network.top[2]]
        fx = neuralnet.evaluate_proposal(data, weights)
        rmse = self.calculate_rmse(fx, y) # Can be replaced by calculate_nmse function for reporting NMSE
        probability = neural_network.softmax(fx)
        loss = 0
        for index_1 in range(y.shape[0]):
            for index_2 in range(y.shape[1]):
                if y[index_1, index_2] == 1:
                    loss += np.log(probability[index_1, index_2] + 0.0001)

        out = np.argmax(fx, axis=1)
        y_out = np.argmax(y, axis=1)
        count = 0
        for index in range(y_out.shape[0]):
            if out[index] == y_out[index]:
                count += 1
        accuracy = float(count)/y_out.shape[0] * 100
        return [loss, rmse, accuracy]

    @staticmethod
    def classification_prior(sigma_squared, weights):
        # h = self.topology[1]  # number hidden neurons
        # d = self.topology[0]  # number input neurons
        part1 = -1 * ((weights.shape[0]) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part1 - part2
        return log_loss

    @staticmethod
    def gaussian_likelihood(neural_network, data, weights, tausq):
        desired = data[:, neural_network.Top[0]: neural_network.Top[0] + neural_network.Top[2]]
        # y_m = data[:, 522:524]
        prediction = neural_network.evaluate_proposal(data, weights)
        # fx_m = Network.denormalize(fx.copy(), [0,1], maxval=[-7299.786516730871000, 4865017.3646842018], minval=[-7695.9387549299299000, 4864745.7450159714])
        # y_m = Network.denormalize(y.copy(), [0,1], maxval=[-7299.786516730871000, 4865017.3646842018], minval=[-7695.9387549299299000, 4864745.7450159714])
        # np.savetxt('y.txt', y, delimiter=',')
        # rmse = self.distance(fx_m, y_m)
        rmse = BayesianTL.calculate_rmse(prediction, desired)
        loss = -0.5 * np.log(2 * np.pi * tausq) - 0.5 * np.square(desired - prediction) / tausq
        return [np.sum(loss), rmse]

    @staticmethod
    def gaussian_prior(sigma_squared, nu_1, nu_2, weights, tausq):
        part1 = -1 * (weights.shape[0] / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def generate_weights(self, w_mean, w_std):
        w_proposal = np.ones(w_mean.shape)
        for index in range(w_mean.shape[0]):
            w_proposal[index] = np.random.normal(w_mean[index], w_std[index], 1)
        return w_proposal

    def transfer(self, weights, eta, likelihood, prior, target_rmse_train, target_rmse_test):
        accept = False
        target_weights_current = weights[0]
        target_rmse_train_current = target_rmse_train
        target_rmse_test_current = target_rmse_test
        target_eta_current = eta[0]

        index =  np.random.uniform(1, self.num_sources, 1).astype('int')
        source_weights_current = weights[index][0]
        source_weights_proposal = weights[self.num_sources + index][0]

        eta_proposal = eta[self.num_sources + index]
        tau_proposal = np.exp(eta_proposal)
        sample_accept, target_rmse_train, target_rmse_test, likelihood, prior = self.evaluate_transfer(self.target, self.target_train_data, self.target_test_data, target_weights_current, source_weights_current, source_weights_proposal, tau_proposal, likelihood, prior)

        if sample_accept:
            target_weights_current = source_weights_proposal
            target_eta_current = eta_proposal
            target_rmse_train_current = target_rmse_train
            target_rmse_test_current = target_rmse_test
            accept = True

        return likelihood, prior, target_weights_current, target_eta_current, target_rmse_train_current, target_rmse_test_current, accept, index


    def evaluate_transfer(self, neural_network, train_data, test_data, target_weights_current, source_weights_current, source_weights_proposal, tau, likelihood, prior):
        accept = False
        [likelihood_proposal, rmse_train] = self.likelihood_function(neural_network, train_data, source_weights_proposal, tau)
        [likelihood_ignore, rmse_test] = self.likelihood_function(neural_network, test_data, source_weights_proposal, tau)
        prior_proposal = self.prior_function(source_weights_proposal, tau)
        difference_likelihood = likelihood_proposal - likelihood
        difference_prior = prior_proposal - prior

        diagmat_size = min(500, target_weights_current.shape[0])
        # diagmat_size = int(0.01 * w_current.shape[0])
        sigma_diagmat = np.zeros((diagmat_size, diagmat_size))
        np.fill_diagonal(sigma_diagmat, 0.02)

        theta_source_current = np.zeros(diagmat_size)
        theta_target_current = np.zeros(diagmat_size)
        theta_source_proposal = np.zeros(diagmat_size)

        indices = np.random.uniform(0, target_weights_current.shape[0], diagmat_size).astype('int')
        for i in range(diagmat_size):
            index = indices[i]
            theta_source_current[i] = source_weights_current[index].copy()
            theta_source_proposal[i] = source_weights_proposal[index].copy()
            theta_target_current[i] = target_weights_current[index].copy()
            i += 1

        transfer_distribution_proposal = multivariate_normal.logpdf(theta_target_current, mean=theta_source_current, cov=sigma_diagmat)
        transfer_distribution_current = multivariate_normal.logpdf(theta_source_proposal, mean=theta_target_current, cov=sigma_diagmat)

        difference_transfer_distribution = transfer_distribution_proposal - transfer_distribution_current

        difference_sum = min(700, difference_likelihood + difference_prior + difference_transfer_distribution)
        mh_transfer_ratio = min(1, np.exp(difference_sum))
        u = random.uniform(0, 1)
        if u < mh_transfer_ratio:
            accept = True
            likelihood = likelihood_proposal
            prior = prior_proposal

        return accept, rmse_train, rmse_test, likelihood, prior


    def acceptance_probability(self, neural_network, train_data, test_data, weights, tau, likelihood, prior):
        accept = False
        [likelihood_proposal, rmse_train] = self.likelihood_function(neural_network, train_data, weights, tau)
        [likelihood_ignore, rmse_test] = self.likelihood_function(neural_network, test_data, weights, tau)
        prior_proposal = self.prior_function(weights, tau)  # takes care of the gradients
        difference_likelihood = likelihood_proposal - likelihood
        difference_prior = prior_proposal - prior
        difference_sum = min(700, difference_likelihood + difference_prior)
        mh_ratio = min(1, np.exp(difference_sum))
        u = random.uniform(0, 1)
        if u < mh_ratio:
            accept = True
            likelihood = likelihood_proposal
            prior = prior_proposal
        return accept, rmse_train, rmse_test, likelihood, prior

    def mcmc_sampler(self, source_weights_initial, target_weights_initial, stdscr, save_knowledge=False, transfer=True, transfer_coefficient=0.01):

        # To save weights for plotting the distributions later
        weights_saved = np.zeros(self.num_sources + 2)
        weights_file = open(self.directory+'/weights.csv', 'w')
        weight_index = 1 # Index of the weight to save

        source_train_rmse_file = open(self.directory+'/source_train_rmse.csv', 'w')
        source_test_rmse_file = open(self.directory+'/source_test_rmse.csv', 'w')

        target_train_rmse_file = open(self.directory+'/target_train_rmse.csv', 'w')
        target_test_rmse_file = open(self.directory+'/target_test_rmse.csv', 'w')

        target_trf_train_rmse_file = open(self.directory+'/target_trf_train_rmse.csv', 'w')
        target_trf_test_rmse_file = open(self.directory+'/target_trf_test_rmse.csv', 'w')

        # ------------------- initialize MCMC
        global start
        start = time.time()

        source_eta = np.zeros((self.num_sources))
        source_tau_proposal = np.zeros((self.num_sources))
        source_prior = np.zeros((self.num_sources))
        source_likelihood = np.zeros((self.num_sources))
        source_likelihood_proposal = np.zeros((self.num_sources))
        source_rmse_train = np.zeros((self.num_sources))
        source_rmse_test = np.zeros((self.num_sources))
        source_rmse_train_sample = np.zeros(source_rmse_train.shape)
        source_rmse_test_sample = np.zeros(source_rmse_test.shape)
        source_train_size = np.zeros((self.num_sources))
        source_test_size = np.zeros((self.num_sources))
        source_weights_current = np.zeros((self.num_sources, self.source_wsize))
        source_weights_proposal = np.zeros((self.num_sources, self.source_wsize))

        source_prediction_train = []
        source_prediction_test = []

        self.weights_stepsize = 0.02  # defines how much variation you need in changes to w
        self.eta_stepsize = 0.01
        self.sigma_squared = 25
        self.nu_1 = 0
        self.nu_2 = 0

        source_y_train = []
        source_y_test = []


        for index in range(self.num_sources):
            source_train_size[index] = self.source_train_data[index].shape[0]
            source_test_size[index] = self.source_test_data[index].shape[0]
            source_y_test.append(self.source_test_data[index][:, self.source_topology[0]: self.source_topology[0] + self.source_topology[2]])
            source_y_train.append(self.source_train_data[index][:, self.source_topology[0]:self.source_topology[0] + self.source_topology[2]])
            source_weights_current[index] = source_weights_initial
            source_weights_proposal[index] = source_weights_initial
            source_prediction_train.append(self.sources[index].evaluate_proposal(self.source_train_data[index], source_weights_current[index]))
            source_prediction_test.append(self.sources[index].evaluate_proposal(self.source_test_data[index], source_weights_current[index]))
            source_eta[index] = np.log(np.var(source_prediction_train[index] - source_y_train[index]))
            source_tau_proposal[index] = np.exp(source_eta[index])
            source_prior[index] = self.prior_function(source_weights_current[index], source_tau_proposal[index])  # takes care of the gradients
            [source_likelihood[index], source_rmse_train[index]] = self.likelihood_function(self.sources[index], self.source_train_data[index], source_weights_current[index], source_tau_proposal[index])
            source_rmse_test[index] = self.calculate_rmse(source_prediction_test[index], source_y_test[index])


        # pos_w = np.ones((self.samples, self.num_sources, self.wsize))  # posterior of all weights and bias over all samples

        target_train_size = self.target_train_data.shape[0]
        target_test_size = self.target_test_data.shape[0]
        target_y_test = self.target_test_data[:, self.target_topology[0]: self.target_topology[0] + self.target_topology[2]]
        target_y_train = self.target_train_data[:, self.target_topology[0]: self.target_topology[0] + self.target_topology[2]]
        target_weights_current = target_weights_initial
        target_weights_proposal = target_weights_initial
        target_prediction_train = self.target.evaluate_proposal(self.target_train_data, target_weights_current)
        target_prediction_test = self.target.evaluate_proposal(self.target_test_data, target_weights_current)
        target_eta = np.log(np.var(target_prediction_train - target_y_train))
        target_tau_proposal = np.exp(target_eta)
        target_prior = self.prior_function(target_weights_current, target_tau_proposal)
        [target_likelihood, target_rmse_train] = self.likelihood_function(self.target, self.target_train_data, target_weights_current, target_tau_proposal)
        target_rmse_test = self.calculate_rmse(target_prediction_test, target_y_test)


        # Copy target values to target with transfer
        target_trf_weights_current = target_weights_current
        target_trf_eta = target_eta
        target_trf_tau_proposal = target_tau_proposal
        target_trf_prior = target_prior
        target_trf_likelihood = target_likelihood
        target_trf_rmse_train = target_rmse_train
        target_trf_rmse_test = target_rmse_test



        for index in range(self.num_sources):
            source_rmse_train_sample[index] = source_rmse_train[index]
            source_rmse_test_sample[index] = source_rmse_test[index]
        source_rmse_train_prev = source_rmse_train
        source_rmse_test_prev = source_rmse_test

        # save the information
        np.savetxt(source_train_rmse_file, [source_rmse_train_sample])
        np.savetxt(source_test_rmse_file, [source_rmse_test_sample])
        np.savetxt(target_train_rmse_file, [target_rmse_train])
        np.savetxt(target_test_rmse_file, [target_rmse_test])

        # save values into previous variables
        target_rmse_train_prev = target_rmse_train
        target_rmse_test_prev = target_rmse_test

        np.savetxt(target_trf_train_rmse_file, [target_trf_rmse_train])
        np.savetxt(target_trf_test_rmse_file, [target_trf_rmse_test])
        # save values into previous variables
        target_trf_rmse_train_prev = target_trf_rmse_train
        target_trf_rmse_test_prev = target_trf_rmse_test

        source_num_accept = np.zeros((self.num_sources))
        target_num_accept = 0
        target_trf_num_accept = 0
        target_trf_num_accept = 0
        num_transfer_accepted = 0
        num_transfer_attempts = 0

        # Add weights that are to be saved
        for index in range(self.num_sources):
            weights_saved[index] = source_weights_current[index, weight_index]
        weights_saved[self.num_sources] = target_weights_current[weight_index]
        weights_saved[self.num_sources + 1] = target_trf_weights_current[weight_index]

        source_prior_proposal = np.zeros((self.num_sources))
        transfer_interval = int( transfer_coefficient * self.num_samples )

        last_transfer_sample  = 0
        last_transfer_rmse = 0
        source_index = None

        for sample in range(self.num_samples - 1):

            source_weights_proposal = source_weights_current + np.random.normal(0, self.weights_stepsize, self.source_wsize)
            target_weights_proposal = target_weights_current + np.random.normal(0, self.weights_stepsize, self.target_wsize)

            source_eta_proposal = source_eta + np.random.normal(0, self.eta_stepsize, 1)
            target_eta_proposal = target_eta + np.random.normal(0, self.eta_stepsize, 1)

            source_tau_proposal = np.exp(source_eta_proposal)
            target_tau_proposal = np.exp(target_eta_proposal)

            if transfer == True:
                target_trf_weights_proposal = target_trf_weights_current + np.random.normal(0, self.weights_stepsize, self.target_wsize)
                target_trf_eta_proposal = target_trf_eta + np.random.normal(0, self.eta_stepsize, 1)
                target_trf_tau_proposal = np.exp(target_trf_eta_proposal)


            # Check MH-acceptance probability for all source tasks
            for index in range(self.num_sources):
                accept, source_rmse_train[index], source_rmse_test[index], source_likelihood[index], source_prior[index] = self.acceptance_probability(self.sources[index], self.source_train_data[index], self.source_test_data[index], source_weights_proposal[index], source_tau_proposal[index], source_likelihood[index], source_prior[index])
                if accept:
                    source_num_accept[index] += 1
                    source_weights_current[index] = source_weights_proposal[index]
                    source_eta[index] = source_eta_proposal[index]
                    source_rmse_train_sample[index] = source_rmse_train[index]
                    source_rmse_test_sample[index] = source_rmse_test[index]
                    source_rmse_train_prev[index] = source_rmse_train[index]
                    source_rmse_test_prev[index] = source_rmse_test[index]
                else:
                    source_rmse_train_sample[index] = source_rmse_train_prev[index]
                    source_rmse_test_sample[index] = source_rmse_test_prev[index]

            if save_knowledge:
                np.savetxt(source_train_rmse_file, [source_rmse_train_sample])
                np.savetxt(source_test_rmse_file, [source_rmse_test_sample])
                weights_saved[: self.num_sources] = source_weights_current[:, weight_index]

            # Check MH-acceptance probability for target task
            accept, target_rmse_train, target_rmse_test, target_likelihood, target_prior = self.acceptance_probability(self.target,                                                     self.target_train_data, self.target_test_data, target_weights_proposal, target_tau_proposal, target_likelihood, target_prior)

            if accept:
                target_num_accept += 1
                target_weights_current = target_weights_proposal
                target_eta = target_eta_proposal
                target_rmse_train_prev = target_rmse_train
                target_rmse_test_prev = target_rmse_test

            if save_knowledge:
                np.savetxt(target_train_rmse_file, [target_rmse_train_prev])
                np.savetxt(target_test_rmse_file, [target_rmse_test_prev])
                weights_saved[self.num_sources] = target_weights_current[weight_index]


            # If transfer is True, evaluate proposal for target task with transfer
            if transfer == True:
                if sample != 0 and sample % transfer_interval == 0:
                    accept = False
                    num_transfer_attempts += 1
                    last_transfer_sample = sample
                    weights_stack = np.vstack([target_trf_weights_current, source_weights_current, source_weights_proposal])
                    source_eta = source_eta.reshape((self.num_sources,1))
                    source_eta_proposal = source_eta_proposal.reshape((self.num_sources,1))
                    eta_stack = np.vstack([target_trf_eta, source_eta, source_eta_proposal])
                    target_trf_likelihood, target_trf_prior, target_trf_weights_current, target_trf_eta, target_trf_rmse_train_prev, target_trf_rmse_test_prev, accept, transfer_index = self.transfer(weights_stack.copy(), eta_stack.copy(), target_trf_likelihood, target_trf_prior, target_trf_rmse_train_prev, target_trf_rmse_test_prev)

                    if accept:
                        target_trf_num_accept = sample
                        num_transfer_accepted += 1
                        last_transfer_rmse = target_trf_rmse_train_prev
                        source_index = transfer_index

                else:
                    accept, target_trf_rmse_train, target_trf_rmse_test, target_trf_likelihood, target_trf_prior = self.acceptance_probability(self.target, self.target_train_data, self.target_test_data, target_trf_weights_proposal, target_trf_tau_proposal, target_trf_likelihood, target_trf_prior)

                    if accept:
                        target_trf_num_accept += 1
                        target_trf_weights_current = target_trf_weights_proposal
                        target_trf_eta = target_trf_eta_proposal

                        # save values into previous variables
                        target_trf_rmse_train_prev = target_trf_rmse_train
                        target_trf_rmse_test_prev = target_trf_rmse_test

                if save_knowledge:
                    np.savetxt(target_trf_train_rmse_file, [target_trf_rmse_train_prev])
                    np.savetxt(target_trf_test_rmse_file, [target_trf_rmse_test_prev])
                    weights_saved[self.num_sources + 1] = target_trf_weights_current[weight_index]
                    np.savetxt(weights_file, [weights_saved], delimiter=',')

            elapsed_time = convert_time(time.time() - start)
            self.report_progress(stdscr, sample, elapsed_time, source_rmse_train_sample, source_rmse_test_sample, target_rmse_train_prev, target_rmse_test_prev, target_trf_rmse_train_prev, target_trf_rmse_test_prev, last_transfer_sample, last_transfer_rmse, source_index, num_transfer_accepted)

        accept_ratio_target = np.array([target_num_accept, target_trf_num_accept]) / float(self.num_samples) * 100
        elapsed_time = time.time() - start
        stdscr.clear()
        stdscr.refresh()
        stdscr.addstr(0 ,0 , r"Sampling Done!, {} % samples were accepted, Total Time: {}".format(accept_ratio_target, elapsed_time))

        accept_ratio = source_num_accept / (self.num_samples * 1.0) * 100
        transfer_ratio = num_transfer_accepted / num_transfer_attempts * 100

        with open(self.directory+"/ratios.txt", 'w') as accept_ratios_file:
            for ratio in accept_ratio:
                accept_ratios_file.write(str(ratio) + ' ')
            for ratio in accept_ratio_target:
                accept_ratios_file.write(str(ratio) + ' ')
            accept_ratios_file.write(str(transfer_ratio) + ' ')


        # Close the files
        source_train_rmse_file.close()
        source_test_rmse_file.close()
        target_train_rmse_file.close()
        target_test_rmse_file.close()
        target_trf_train_rmse_file.close()
        target_trf_test_rmse_file.close()
        weights_file.close()

        return (accept_ratio, transfer_ratio)



    def get_rmse(self):
        self.source_rmse_train = np.genfromtxt(self.directory+'/source_train_rmse.csv')
        self.source_rmse_test = np.genfromtxt(self.directory+'/source_test_rmse.csv')
        if self.num_sources == 1:
            self.source_rmse_test = self.source_rmse_test.reshape((self.source_rmse_test.shape[0], 1))
            self.source_rmse_train = self.source_rmse_train.reshape((self.source_rmse_train.shape[0], 1))
        self.target_rmse_train = np.genfromtxt(self.directory+'/target_train_rmse.csv')
        self.target_rmse_test = np.genfromtxt(self.directory+'/target_test_rmse.csv')
        self.target_trf_rmse_train = np.genfromtxt(self.directory+'/target_trf_train_rmse.csv')
        self.target_trf_rmse_test = np.genfromtxt(self.directory+'/target_trf_test_rmse.csv')
        # print self.source_rmse_test.shape


    def display_rmse(self):
        burnin = 0.1 * self.num_samples  # use post burn in samples
        self.get_rmse()

        rmse_tr = [0 for index in range(self.num_sources)]
        rmsetr_std = [0 for index in range(self.num_sources)]
        rmse_tes = [0 for index in range(self.num_sources)]
        rmsetest_std = [0 for index in range(self.num_sources)]

        for index in range(self.num_sources):
            rmse_tr[index] = np.mean(self.source_rmse_train[int(burnin):, index])
            rmsetr_std[index] = np.std(self.source_rmse_train[int(burnin):, index])

            rmse_tes[index] = np.mean(self.source_rmse_test[int(burnin):, index])
            rmsetest_std[index] = np.std(self.source_rmse_test[int(burnin):, index])

        rmse_target_train = np.mean(self.target_rmse_train[int(burnin):])
        rmsetarget_std_train = np.std(self.target_rmse_train[int(burnin):])

        rmse_target_test = np.mean(self.target_rmse_test[int(burnin):])
        rmsetarget_std_test = np.std(self.target_rmse_test[int(burnin):])


        rmse_target_train_trf = np.mean(self.target_trf_rmse_train[int(burnin):])
        rmsetarget_std_train_trf = np.std(self.target_trf_rmse_train[int(burnin):])

        rmse_target_test_trf = np.mean(self.target_trf_rmse_test[int(burnin):])
        rmsetarget_std_test_trf = np.std(self.target_trf_rmse_test[int(burnin):])

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

        burnin = int(0.1 * self.num_samples)

        for index in range(self.num_sources):
            ax = plt.subplot(111)
            x = np.array(np.arange(burnin, self.samples))
            plt.plot(x, self.source_rmse_train[burnin: , index], '.' , label="train")
            plt.plot(x, self.source_rmse_test[burnin: , index], '.' , label="test")
            plt.legend()
            plt.xlabel('Samples')
            plt.ylabel('RMSE')
            plt.title(dataset+' Source '+str(index+1)+' RMSE')
            plt.savefig(self.directory+'/results/rmse-source-'+str(index+1)+'.png')
            plt.clf()

        ax = plt.subplot(111)
        x = np.array(np.arange(burnin, self.samples))
        plt.plot(x, self.target_rmse_train[burnin: ], '.' , label="no-transfer")
        plt.plot(x, self.target_trf_rmse_train[burnin: ], '.' , label="transfer")
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('RMSE')
        plt.title(dataset+' Train RMSE')
        plt.savefig(self.directory+'/results/rmse-target-train-mcmc.png')
        plt.clf()


        ax = plt.subplot(111)
        plt.plot(x, self.target_rmse_test[burnin: ], '.' , label="no-transfer")
        plt.plot(x, self.target_trf_rmse_test[burnin: ], '.' , label="transfer")
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
    num_sources = [1, 1, 1, 5]
    type = {0:'classification', 1:'regression', 2:'regression', 3:'regression'}
    num_samples = [8000, 100, 4000, 8000]

    problem = 1
    problem_type = type[problem]
    topology = [input[problem], hidden[problem], output[problem]]
    problem_name = name[problem]

    start = None
    #--------------------------------------------- Train for the source task -------------------------------------------

    stdscr = None
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()

    try:
        # stdscr.clear()
        # target_train_data = np.genfromtxt('../datasets/WineQualityDataset/preprocess/winequality-red-train.csv', delimiter=',')
        # target_test_data = np.genfromtxt('../datasets/WineQualityDataset/preprocess/winequality-red-test.csv', delimiter=',')
        target_train_data = np.genfromtxt('../datasets/UJIndoorLoc/targetData/0train.csv', delimiter=',')[:, :-2]
        target_test_data = np.genfromtxt('../datasets/UJIndoorLoc/targetData/0test.csv', delimiter=',')[:, :-2]
        # target_train_data = np.genfromtxt('../datasets/synthetic_data/target_train.csv', delimiter=',')
        # target_test_data = np.genfromtxt('../datasets/synthetic_data/target_test.csv', delimiter=',')
        # target_train_data = np.genfromtxt('../datasets/Sarcos/target_train.csv', delimiter=',')
        # target_test_data = np.genfromtxt('../datasets/Sarcos/target_test.csv', delimiter=',')

        train_data = []
        test_data = []
        for index in range(num_sources[problem]):
            # train_data.append(np.genfromtxt('../datasets/WineQualityDataset/preprocess/winequality-white-train.csv', delimiter=','))
            # test_data.append(np.genfromtxt('../datasets/WineQualityDataset/preprocess/winequality-red-test.csv', delimiter=','))
            train_data.append(np.genfromtxt('../datasets/UJIndoorLoc/sourceData/'+str(index)+'train.csv', delimiter=',')[:, :-2])
            test_data.append(np.genfromtxt('../datasets/UJIndoorLoc/sourceData/'+str(index)+'test.csv', delimiter=',')[:, :-2])
            # train_data.append(np.genfromtxt('../datasets/synthetic_data/source'+str(i+1)+'.csv', delimiter=','))
            # test_data.append(np.genfromtxt('../datasets/synthetic_data/target_test.csv', delimiter=','))
            # train_data.append(np.genfromtxt('../datasets/Sarcos/source.csv', delimiter=','))
            # test_data.append(np.genfromtxt('../datasets/Sarcos/target_test.csv', delimiter=','))
            pass

        # stdscr.clear()
        random.seed(time.time())

        mcmc_task = BayesianTL(num_samples[problem], num_sources[problem], train_data, test_data, target_train_data, target_test_data, topology,  directory=problem_name, type=problem_type)  # declare class

        # generate random weights
        w_random = np.random.randn(mcmc_task.source_wsize)
        w_random_target = np.random.randn(mcmc_task.target_wsize)

        # start sampling
        accept_ratio, transfer_ratio = mcmc_task.mcmc_sampler(w_random, w_random_target, save_knowledge=True, stdscr=stdscr, transfer=True, transfer_coefficient=0.05)

        # display train and test accuracies
        mcmc_task.display_rmse()

        # Plot the accuracies and rmse
        mcmc_task.plot_rmse(problem_name)

    finally:
        curses.echo()
        curses.nocbreak()
        curses.endwin()
        pass
