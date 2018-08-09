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

# --------------------------------------------- Basic Neural Network Class ---------------------------------------------

class Network(object):

    def __init__(self, Topo, learn_rate = 0.5, alpha = 0.1):
        self.Top = Topo  # NN topology [input, hidden, output]
        np.random.seed(int(time.time()))
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
    def scale_data(data, maxout=1, minout=0, maxin=1, minin=0):
        attribute = data[:]
        attribute = minout + (attribute - minin)*((maxout - minout)/(maxin - minin))
        return attribute

    @staticmethod
    def denormalize(data, indices, maxval, minval):
        for i in range(len(indices)):
            index = indices[i]
            attribute = data[:, index]
            attribute = Network.scale_data(attribute, maxout=maxval[i], minout=minval[i], maxin=1, minin=0)
            data[:, index] = attribute
        return data

    @staticmethod
    def softmax(fx):
        ex = np.exp(fx)
        sum_ex = np.sum(ex, axis = 1)
        sum_ex = np.multiply(np.ones(ex.shape), sum_ex[:, np.newaxis])
        prob = np.divide(ex, sum_ex)
        return prob

    def evaluate_proposal(self, data, w):  # BP with SGD (Stocastic BP)
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


# ---------------------------------------------- Bayesian Transfer Learning Class ------------------------------------------------
class BayesianTL(object):
    def __init__(self, num_samples, num_sources, source_train_data, source_test_data, target_train_data, target_test_data, topology, directory, problem_type='regression'):
        self.num_samples = num_samples
        self.topology = topology
        self.source_train_data = source_train_data  #
        self.source_test_data = source_test_data
        self.target_train_data = target_train_data
        self.target_test_data = target_test_data
        self.num_sources = num_sources
        self.problem_type = problem_type
        self.directory = directory
        self.source_wsize = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        self.neural_network = Network(topology)
        self.join_data()
        self.initialize_mu()
        self.create_directory(self.directory)

    # ----------------------------------------------------------------------------------------------------------------------------
    @staticmethod
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

    @staticmethod
    def joint_prior_density(weights, mu, nu_squared):
        n = phi.shape[0]
        part_1 = -np.sum(np.square(weights - mu)) / (2 * nu_squared)
        part_2 = -n/2 * np.log(2 * np.pi * nu_squared)
        loss = np.sum(np.log( 1 / (1 +  np.square((weights - mu))))) - part_1 - part_2
        return loss

    @staticmethod
    def create_directory(directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)

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
                    loss += np.log(probability[index_1, index_2])
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
        part1 = -1 * ((weights.shape[0]) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part1 - part2
        return log_loss

    @staticmethod
    def gaussian_likelihood(neural_network, data, weights, tausq):
        desired = data[:, neural_network.Top[0]: neural_network.Top[0] + neural_network.Top[2]]
        prediction = neural_network.evaluate_proposal(data, weights)
        rmse = BayesianTL.calculate_rmse(prediction, desired)
        loss = -0.5 * np.log(2 * np.pi * tausq) - 0.5 * np.square(desired - prediction) / tausq
        return [np.sum(loss), rmse]

    @staticmethod
    def gaussian_prior(sigma_squared, nu_1, nu_2, weights, tausq):
        part1 = -1 * (weights.shape[0] / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def likelihood_function(self, neural_network, data, weights, tau):
        if self.problem_type == 'regression':
            likelihood, rmse = self.gaussian_likelihood(neural_network, data, weights, tau)
        elif self.problem_type == 'classification':
            likelihood, rmse, accuracy = self.multinomial_likelihood(neural_network, data, weights)
        return likelihood, rmse

    def prior_function(self, weights, tau):
        if self.type == 'regression':
            loss = self.gaussian_prior(self.sigma_squared, self.nu_1, self.nu_2, weights, tau)
        elif self.type == 'classification':
            loss = self.classification_prior(self.sigma_squared, weights)
        return loss

    def report_progress(self, stdscr, sample_count, elapsed, rmse_train_source, rmse_test_source, rmse_train_target, rmse_test_target, rmse_train_target_trf, rmse_test_target_trf, last_transfer_sample, last_transfer_rmse, source_index, last_transfer_accepted):
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
        stdscr.addstr(i + 2, 5, "Last transfer attempt: {} Last transfered rmse: {:.4f} Source index: {} Last transfer accepted: {} ".format(last_transfer_sample, last_transfer_rmse, source_index, last_transfer_accepted))
        stdscr.refresh()

    def join_data(self):
        train_data = self.target_train_data
        test_data = self.target_test_data
        for index in range(self.num_sources):
            train_data = np.vstack([train_data, self.source_train_data[index]])
            test_data = np.vstack([test_data, self.source_test_data[index]])
        self.joint_train_data = train_data
        self.joint_test_data = test_data

    def evaluate_proposal(self, neural_network, train_data, test_data, weights_proposal, tau_proposal, likelihood_current, prior_current, mu=None):
            accept = False
            likelihood_ignore, rmse_test_proposal = self.likelihood_function(neural_network, test_data, weights, tau)
            likelihood_proposal, rmse_train_proposal = self.likelihood_function(neural_network, train_data, weights, tau)
            if mu == None:
                prior_proposal = self.prior_function(weights_proposal, tau_proposal)
            else:
                prior_proposal = self.joint_prior_distribution(weights_proposal, mu, self.nu_squared)
            difference_likelihood = likelihood_proposal - likelihood_current
            difference_prior = prior_proposal - prior_current
            mh_ratio = min(1, np.exp(min(709, difference_likelihood + difference_prior)))
            u = np.random.uniform(0,1)
            if u < mh_ratio:
                accept = True
                likelihood_current = likelihood_proposal
                prior_proposal = prior_current
            return accept, rmse_train_proposal, rmse_test_proposal, likelihood_current, prior_current

    def mcmc_sampler(self, stdscr, save_knowledge=False):

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
        source_rmse_train_current = source_rmse_train
        source_rmse_test_current = source_rmse_test

        # save the information
        np.savetxt(source_train_rmse_file, [source_rmse_train_sample])
        np.savetxt(source_test_rmse_file, [source_rmse_test_sample])
        np.savetxt(target_train_rmse_file, [target_rmse_train])
        np.savetxt(target_test_rmse_file, [target_rmse_test])

        # save values into previous variables
        target_rmse_train_current = target_rmse_train
        target_rmse_test_current = target_rmse_test

        np.savetxt(target_trf_train_rmse_file, [target_trf_rmse_train])
        np.savetxt(target_trf_test_rmse_file, [target_trf_rmse_test])
        # save values into previous variables
        target_trf_rmse_train_current = target_trf_rmse_train
        target_trf_rmse_test_current = target_trf_rmse_test

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

        last_transfer_sample  = None
        last_transfer_rmse = float()
        source_index = None
        last_transfer_accepted = None

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
                    source_rmse_train_current[index] = source_rmse_train[index]
                    source_rmse_test_current[index] = source_rmse_test[index]
                else:
                    source_rmse_train_sample[index] = source_rmse_train_current[index]
                    source_rmse_test_sample[index] = source_rmse_test_current[index]

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
                target_rmse_train_current = target_rmse_train
                target_rmse_test_current = target_rmse_test

            if save_knowledge:
                np.savetxt(target_train_rmse_file, [target_rmse_train_current])
                np.savetxt(target_test_rmse_file, [target_rmse_test_current])
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
                    target_trf_likelihood, target_trf_prior, target_trf_weights_current, target_trf_eta, target_trf_rmse_train_current, target_trf_rmse_test_current, accept, transfer_index = self.transfer(weights_stack.copy(), eta_stack.copy(), target_trf_likelihood, target_trf_prior, target_trf_rmse_train_current, target_trf_rmse_test_current)

                    if accept:
                        target_trf_num_accept += 1
                        num_transfer_accepted += 1
                        last_transfer_rmse = target_trf_rmse_train_current
                        source_index = transfer_index
                        last_transfer_accepted = sample

                else:
                    accept, target_trf_rmse_train, target_trf_rmse_test, target_trf_likelihood, target_trf_prior = self.acceptance_probability(self.target, self.target_train_data, self.target_test_data, target_trf_weights_proposal, target_trf_tau_proposal, target_trf_likelihood, target_trf_prior)

                    if accept:
                        target_trf_num_accept += 1
                        target_trf_weights_current = target_trf_weights_proposal
                        target_trf_eta = target_trf_eta_proposal

                        # save values into previous variables
                        target_trf_rmse_train_current = target_trf_rmse_train
                        target_trf_rmse_test_current = target_trf_rmse_test

                if save_knowledge:
                    np.savetxt(target_trf_train_rmse_file, [target_trf_rmse_train_current])
                    np.savetxt(target_trf_test_rmse_file, [target_trf_rmse_test_current])
                    weights_saved[self.num_sources + 1] = target_trf_weights_current[weight_index]
                    np.savetxt(weights_file, [weights_saved], delimiter=',')

            elapsed_time = BayesianTL.convert_time(time.time() - start)
            self.report_progress(stdscr, sample, elapsed_time, source_rmse_train_sample, source_rmse_test_sample, target_rmse_train_current, target_rmse_test_current, target_trf_rmse_train_current, target_trf_rmse_test_current, last_transfer_sample, last_transfer_rmse, source_index, last_transfer_accepted)

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
    # stdscr = curses.initscr()
    # curses.noecho()
    # curses.cbreak()

    try:
        # stdscr.clear()
        # target_train_data = np.genfromtxt('../datasets/WineQualityDataset/preprocess/winequality-red-train.csv', delimiter=',')
        # target_test_data = np.genfromtxt('../datasets/WineQualityDataset/preprocess/winequality-red-test.csv', delimiter=',')
        target_train_data = np.genfromtxt('../datasets/UJIndoorLoc/targetData/0train.csv', delimiter=',')[:, :-2]
        target_test_data = np.genfromtxt('../datasets/UJIndoorLoc/targetData/0test.csv', delimiter=',')[:, :-2]
        print('target data shape:', target_train_data.shape)
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
            print('source data shape:', train_data[index].shape)
            # train_data.append(np.genfromtxt('../datasets/synthetic_data/source'+str(i+1)+'.csv', delimiter=','))
            # test_data.append(np.genfromtxt('../datasets/synthetic_data/target_test.csv', delimiter=','))
            # train_data.append(np.genfromtxt('../datasets/Sarcos/source.csv', delimiter=','))
            # test_data.append(np.genfromtxt('../datasets/Sarcos/target_test.csv', delimiter=','))
            pass

        # stdscr.clear()
        random.seed(time.time())

        bayesTL = BayesianTL(num_samples[problem], num_sources[problem], train_data, test_data, target_train_data, target_test_data, topology,  directory=problem_name, type=problem_type)  # declare class

        print('joint train data size:',bayesTL.joint_train_data.shape)

        # generate random weights
        w_random = np.random.randn(bayesTL.source_wsize)
        w_random_target = np.random.randn(bayesTL.target_wsize)

        # display train and test accuracies
        # bayesTL.display_rmse()

        # Plot the accuracies and rmse
        # bayesTL.plot_rmse(problem_name)

    finally:
        # curses.echo()
        # curses.nocbreak()
        # curses.endwin()
        pass
