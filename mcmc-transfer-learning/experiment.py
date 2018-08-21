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

class Experiment(object):
    def __init__(self, num_samples = 2000):
        self.mu = 0
        self.n_1 = 80
        self.n_2 = 20
        self.sigma_mu_sq = 25
        self.sigma_delta_sq = 1
        self.tau_sq = 1
        self.s_delta_sq = 0.25
        self.s_mu_sq = 0.25
        self.delta_1, self.delta_2 = np.random.normal(self.mu, self.sigma_delta_sq, 2)
        self.y_1 = np.random.normal(self.delta_1, self.tau_sq, self.n_1)
        self.y_2 = np.random.normal(self.delta_2, self.tau_sq, self.n_2)
        self.Y = np.concatenate((self.y_1, self.y_2), axis=0)
        self.f_1 = self.func(self.delta_1)
        self.f_2 = self.func(self.delta_2)
        self.F = self.func(self.mu)
        self.num_samples = num_samples
        self.delta_file_1 = 'delta_1.csv'
        self.delta_file_2 = 'delta_2.csv'
        self.mu_file = 'mu.csv'
        self.joint_file ='joint_sampling.csv'

    @staticmethod
    def func(delta):
        fx =  1 / (1 + np.exp(delta))
        return fx

    def task_acceptance_ratio(self, y, fx_c, fx_p):
        alpha = np.exp(-0.5/self.tau_sq * (np.sum(np.square(y - fx_p)) - np.sum(np.square(y - fx_c))))
        return alpha

    def joint_likelihood(self, y_1, f_1, y_2, f_2):
        loss = float(self.n_1 + self.n_2)/2*np.log(2*np.pi*self.tau_sq) - (np.sum(np.square(y_1 - f_1)) + np.sum(np.square(y_2 - f_2)))/(2 * self.tau_sq)
        return loss

    def joint_prior(self, delta_1, delta_2, mu):
        loss =  -np.log(2*np.pi*self.s_delta_sq) - (np.sum(np.square(delta_1 - mu)) + np.sum(np.square(delta_2 - mu)))/(2*self.s_delta_sq)
        return loss

    def joint_acceptance_ratio(self, delta_1_c, delta_2_c, delta_1_p, delta_2_p, y_1, y_2, f_delta_1_c, f_delta_1_p, f_delta_2_c, f_delta_2_p, mu_c):
        diff_likelihood = self.joint_likelihood(y_1, f_delta_1_p, y_2, f_delta_2_p) - self.joint_likelihood(y_1, f_delta_1_c, y_2, f_delta_2_c)
        diff_prior = self.joint_prior(delta_1_p, delta_2_p, mu_c) - self.joint_prior(delta_1_c, delta_2_c, mu_c)
        alpha = min(1, np.exp(diff_likelihood + diff_prior))
        return alpha

    # task equals 1, 2 or 3
    def task_sampler(self, task=1):
        if task not in [1, 2, 3]:
            raise ValueError('task can only take values 1, 2 or 3')
        vars = {1: ('source', self.delta_1, self.y_1, self.f_1, self.delta_file_1, self.s_delta_sq), 2: ('target', self.delta_2, self.y_2, self.f_2, self.delta_file_2, self.s_delta_sq), 3: ('joint', self.mu, self.Y, self.F, self.mu_file, self.s_mu_sq)}
        name, param_c, y, f_c, path, step_size = vars[task]
        file = open(path, 'w')
        print("Starting sampling for " + name + "...")
        for sample in range(self.num_samples):
            param_p = param_c + np.random.normal(0, step_size)
            f_p = self.func(param_p)
            alpha = self.task_acceptance_ratio(y, f_c, f_p)
            u = np.random.uniform(0, 1)
            if u < alpha:
                param_c = param_p
                f_c = f_p
            np.savetxt(file, np.array([param_c]), delimiter =',')
            print(name, param_c)
        file.close()
        return

    def joint_sampler(self):
        mu = self.mu
        delta_1_c, delta_2_c = self.delta_1, self.delta_2
        f_delta_1_c = self.f_1
        f_delta_2_c = self.f_2
        file = open(self.joint_file, 'w')
        for sample in range(1, self.num_samples):
            # Propose delta_1 and delta_2
            delta_1_p = delta_1_c + np.random.normal(0, self.s_delta_sq)
            delta_2_p = delta_2_c + np.random.normal(0, self.s_delta_sq)
            # Evaluate function f
            f_delta_1_p = self.func(delta_1_p)
            f_delta_2_p = self.func(delta_2_p)
            # get joint acceptance ratio value
            alpha = self.joint_acceptance_ratio(delta_1_c, delta_2_c, delta_1_p, delta_2_p, self.y_1, self.y_2, f_delta_1_c, f_delta_1_p, f_delta_2_c, f_delta_2_p, mu)
            u = np.random.uniform(0, 1)
            if u < alpha:
                delta_1_c = delta_1_p
                delta_2_c = delta_2_p
                f_delta_1_c = f_delta_1_p
                f_delta_2_c = f_delta_2_p
            weights = np.array([delta_1_c, delta_2_c, mu]).reshape(1, 3)
            np.savetxt(file, weights, delimiter=',')
            print('weights: ', weights)
            mu = np.random.normal((delta_1_c + delta_2_c)/2, self.sigma_mu_sq/2)
        file.close()
        return

    def plot_histogram(self):
        delta_1 = np.genfromtxt(self.delta_file_1, delimiter=',')
        delta_2 = np.genfromtxt(self.delta_file_2, delimiter=',')
        mu = np.genfromtxt(self.mu_file, delimiter=',')
        ax = plt.subplot(111)
        burnin = int(0.1 * self.num_samples)
        plt.hist(mu[burnin:], bins=45, alpha=0.7, facecolor='sandybrown', edgecolor='r', label='mu', density=True)
        plt.hist(delta_1[burnin:], bins=70, alpha=0.7, facecolor='C0', edgecolor='b', label='source', density=True)
        plt.hist(delta_2[burnin:], bins=50, alpha=0.7, facecolor='C8', edgecolor='g', label='target', density=True)
        plt.legend()
        plt.title('Weight Density plot')
        plt.xlabel('Parameter value')
        plt.ylabel('Density')
        plt.savefig('weight.png')
        plt.clf()

        weights = np.genfromtxt(self.joint_file, delimiter=',')
        ax = plt.subplot(111)
        burnin = int(0.1 * self.num_samples)
        plt.hist(weights[burnin:, 0], bins=45, alpha=0.7, facecolor='sandybrown', edgecolor='r', label='source', density=True)
        plt.hist(weights[burnin:, 1], bins=70, alpha=0.7, facecolor='C0', edgecolor='b', label='target', density=True)
        plt.hist(weights[burnin:, 2], bins=50, alpha=0.7, facecolor='C8', edgecolor='g', label='mu', density=True)
        plt.legend()
        plt.title('Weight Density plot')
        plt.xlabel('Parameter value')
        plt.ylabel('Density')
        plt.savefig('joint_weight.png')
        plt.clf()


if __name__ == '__main__':
    num_samples = 2000
    print("Initializing... ")
    experiment = Experiment(num_samples)
    print("Initialized! Now running sampler..")
    for task in range(1, 4):
        experiment.task_sampler(task)
    print('Starting joint smapling...')
    experiment.joint_sampler()
    experiment.plot_histogram()
