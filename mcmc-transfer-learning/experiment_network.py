# !/usr/bin/python
from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import norm
from scipy.special import gamma
import os
import sys

#----------------------------------------------------- Neural Network Class-------------------------------------------------------
class Network(object):

    def __init__(self, topology, learn_rate = 0.5, alpha = 0.1):
        self.topology = topology  # NN topology [input, hidden, output]
        np.random.seed(int(time.time()))
        self.W1 = np.random.randn(self.topology[0], self.topology[1]) / np.sqrt(self.topology[0])
        self.B1 = np.random.randn(1, self.topology[1]) / np.sqrt(self.topology[1])  # bias first layer
        self.W2 = np.random.randn(self.topology[1], self.topology[2]) / np.sqrt(self.topology[1])
        self.B2 = np.random.randn(1, self.topology[2]) / np.sqrt(self.topology[1])  # bias second layer

        self.hidout = np.zeros((1, self.topology[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.topology[2]))  # output last layer

    @staticmethod
    def sigmoid(x):
        x = x.astype(np.float128)
        return 1 / (1 + np.exp(-x))

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.topology[2]
        return sqerror

    def sampleAD(self, actualout):
        error = np.subtract(self.out, actualout)
        moderror = np.sum(np.abs(error)) / self.topology[2]
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
        w_layer1size = self.topology[0] * self.topology[1]
        w_layer2size = self.topology[1] * self.topology[2]

        w_layer1 = w[0:w_layer1size]

        self.W1 = np.reshape(w_layer1, (self.topology[0], self.topology[1]))
        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.topology[1], self.topology[2]))
        self.B1 = w[w_layer1size + w_layer2size :w_layer1size + w_layer2size + self.topology[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.topology[1] :w_layer1size + w_layer2size + self.topology[1] + self.topology[2]]

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

    def generate_output(self, data, w):  # BP with SGD (Stocastic BP)
        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]
        Input = np.zeros((1, self.topology[0]))  # temp hold input
        Desired = np.zeros((1, self.topology[2]))
        fx = np.zeros((size,self.topology[2]))
        for i in range(0, size):  # to see what fx is produced by your current weight update
            Input = data[i, 0:self.topology[0]]
            self.ForwardPass(Input)
            fx[i] = self.out
        return fx

#---------------------------------------------------------Experiment Sampler-----------------------------------------------------
class Experiment(object):
    def __init__(self, topology, train_data, test_data, num_samples = 2000):
        self.train_data = train_data
        self.test_data = test_data
        self.topology = topology
        self.neural_network = Network(topology)
        self.num_tasks = 2
        self.w_size = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        self.mu = np.zeros(self.w_size)
        self.num_obv = np.array([train_data[task].shape[0] for task in range(self.num_tasks)])
        self.sigma_mu_sq = 25
        self.sigma_sq = 1
        self.tau_sq = 0.1
        self.step_size = 0.02
        self.delta = np.random.normal(self.mu, np.sqrt(self.sigma_sq), (self.num_tasks, self.w_size))
        self.fx = [self.neural_network.generate_output(self.train_data[task], self.delta[task]) for task in range(self.num_tasks)]
        self.y = [self.train_data[task][:, self.topology[0]: self.topology[0]+self.topology[2]] for task in range(self.num_tasks)]
        self.Y = np.concatenate(self.y, axis=0)
        # self.F = self.neural_network.generate_output(self.Y, self.mu)
        self.num_samples = num_samples
        self.delta_files = ['delta_'+str(task+1)+'.csv' for task in range(self.num_tasks)]
        self.mu_file = 'mu.csv'
        self.joint_file ='joint_sampling_network.csv'

    # def task_acceptance_ratio(self, y, fx_c, fx_p):
    #     alpha = np.exp(-0.5/self.tau_sq * (np.sum(np.square(y - fx_p)) - np.sum(np.square(y - fx_c))))
    #     return alpha

    def joint_likelihood(self, y, f, tau_sq):
        log_likelihood = np.sum(np.array([np.sum(norm.logpdf(f[task], y[task], np.sqrt(tau_sq))) for task in range(self.num_tasks)]))
        return log_likelihood

    def joint_prior(self, delta, mu, sigma_sq):
        log_prior = np.sum(np.array([norm.logpdf(delta[task], mu, np.sqrt(sigma_sq)) for task in range(self.num_tasks)]))
        return log_prior

    def joint_acceptance_ratio(self, delta_c, delta_p, y, f_c, f_p, mu, tau_sq, sigma_sq):
        diff_likelihood = self.joint_likelihood(y, f_p, tau_sq) - self.joint_likelihood(y, f_c, tau_sq)
        print("Likelihood: {}".format(self.joint_likelihood(y, f_c, tau_sq)), end=' ')
        diff_prior = self.joint_prior(delta_p, mu, sigma_sq) - self.joint_prior(delta_c, mu, sigma_sq)
        alpha = min(1, np.exp(min(709, diff_likelihood + diff_prior)))
        return alpha

    # task equals 1, 2 or 3
    # def task_sampler(self, task=1):
    #     if task not in [1, 2, 3]:
    #         raise ValueError('task can only take values 1, 2 or 3')
    #     vars = {1: ('source', self.delta_1, self.y_1, self.f_1, self.delta_file_1, self.s_delta_sq), 2: ('target', self.delta_2, self.y_2, self.f_2, self.delta_file_2, self.s_delta_sq), 3: ('joint', self.mu, self.Y, self.F, self.mu_file, self.s_mu_sq)}
    #     name, param_c, y, f_c, path, step_size = vars[task]
    #     file = open(path, 'w')
    #     print("Starting sampling for " + name + "...")
    #     for sample in range(self.num_samples):
    #         param_p = param_c + np.random.normal(0, step_size)
    #         f_p = self.func(param_p)
    #         alpha = self.task_acceptance_ratio(y, f_c, f_p)
    #         u = np.random.uniform(0, 1)
    #         if u < alpha:
    #             param_c = param_p
    #             f_c = f_p
    #         np.savetxt(file, np.array([param_c]), delimiter =',')
    #         print(name, param_c)
    #     file.close()
    #     return

    def joint_sampler(self):
        sig_a_prior = 2
        sig_b_prior = 2 * sig_a_prior
        tau_a_prior = 2
        tau_b_prior = 2 * tau_a_prior
        tau_sq_c = self.tau_sq
        sigma_sq_c = self.sigma_sq
        mu = self.mu
        delta_c = self.delta
        fx_c = self.fx
        file = open(self.joint_file, 'w')
        for sample in range(1, self.num_samples):
            # Propose delta_1 and delta_2
            delta_p = delta_c + np.random.normal(0, self.step_size, (self.num_tasks, self.w_size))
            # Evaluate function f
            fx_p = [self.neural_network.generate_output(self.train_data[task], delta_p[task]) for task in range(self.num_tasks)]
            # get joint acceptance ratio value
            alpha = self.joint_acceptance_ratio(delta_c, delta_p, self.y, fx_c, fx_p, mu, tau_sq_c, sigma_sq_c)
            u = np.random.uniform(0, 1)
            if u < alpha:
                delta_c = delta_p
                fx_c = fx_p
            # Drawing tau_sq
            tau_a = np.sum(self.num_obv)/2 + tau_a_prior;
            tau_b = tau_b_prior
            tau_b = tau_b + np.sum(np.array([np.sum(np.square(self.y[task] - fx_c[task]))/2 for task in range(self.num_tasks)]))
            tau_sq_c = 1 / np.random.gamma(tau_a, tau_b)
            # Save weights
            weights = np.concatenate((delta_c[:, 10], np.array([mu[10]]))).reshape(1, self.num_tasks + 1)
            np.savetxt(file, weights, delimiter=',')
            print('weights: ', weights, end=' ' )
            # Draw mu
            mu = np.random.normal(delta_c.mean(axis=0), np.sqrt(sigma_sq_c/2))
            # Drawing sigma_sq
            sig_a = self.num_tasks/2 + sig_a_prior
            sig_b = np.sum(np.square(delta_c - mu))/2 + sig_b_prior
            sigma_sq_c = 1 / np.random.gamma(sig_a, sig_b)
            print ("Samples: {}".format(sample))

        file.close()
        return

    def plot_histogram(self):
        # delta_1 = np.genfromtxt(self.delta_file_1, delimiter=',')
        # delta_2 = np.genfromtxt(self.delta_file_2, delimiter=',')
        # mu = np.genfromtxt(self.mu_file, delimiter=',')
        # ax = plt.subplot(111)
        # burnin = int(0.1 * self.num_samples)
        # plt.hist(mu[burnin:], bins=45, alpha=0.7, facecolor='sandybrown', edgecolor='r', label='mu', density=True)
        # plt.hist(delta_1[burnin:], bins=70, alpha=0.7, facecolor='C0', edgecolor='b', label='source', density=True)
        # plt.hist(delta_2[burnin:], bins=50, alpha=0.7, facecolor='C8', edgecolor='g', label='target', density=True)
        # plt.legend()
        # plt.title('Weight Density plot')
        # plt.xlabel('Parameter value')
        # plt.ylabel('Density')
        # plt.savefig('weight.png')
        # plt.clf()

        weights = np.genfromtxt(self.joint_file, delimiter=',')
        burnin = int(0.1 * self.num_samples)
        for task in range(self.num_tasks):
            ax = plt.subplot(self.num_tasks+1, 1, task+1)
            plt.hist(weights[burnin:, task], bins=50, alpha=0.7, facecolor='sandybrown', edgecolor='r', label='task '+str(task+1), density=True)
            plt.legend()
        ax = plt.subplot(self.num_tasks+1, 1, self.num_tasks+1)
        plt.hist(weights[burnin:, 2], bins=50, alpha=0.7, facecolor='C8', edgecolor='g', label='mu', density=True)
        plt.legend()
        plt.xlabel('Parameter value')
        plt.ylabel('Density')
        plt.savefig('joint_weight_network.png')
        plt.clf()


if __name__ == '__main__':
    num_samples = 25000
    topology = [4, 15, 1]
    train_data = []
    test_data = []
    index = 3
    train_data.append(np.genfromtxt('../datasets/synthetic_data/source'+str(index+1)+'.csv', delimiter=','))
    train_data.append(np.genfromtxt('../datasets/synthetic_data/target_train.csv', delimiter=','))

    test_data.append(np.genfromtxt('../datasets/synthetic_data/target_test.csv', delimiter=','))
    test_data.append(np.genfromtxt('../datasets/synthetic_data/target_test.csv', delimiter=','))




    random.seed(time.time())

    print("Initializing... ")
    experiment = Experiment(topology, train_data, test_data, num_samples)
    # print("Initialized! Now running sampler..")
    # for task in range(1, 4):
    #     experiment.task_sampler(task)
    print('Starting joint sampling...')
    experiment.joint_sampler()
    experiment.plot_histogram()
