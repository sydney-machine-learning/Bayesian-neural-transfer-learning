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
        self.num_tasks = 2
        self.mu = 0
        self.num_obv = np.array([80, 20])
        self.sigma_mu_sq = 25
        self.sigma_delta_sq = 1
        self.tau_sq = 0.1
        self.s_delta_sq = 0.05
        self.s_mu_sq = 0.05
        self.delta = np.random.normal(self.mu, np.sqrt(self.sigma_delta_sq), self.num_tasks)
        self.fx = self.func(self.delta)
        self.y = [np.random.normal(self.fx[task], np.sqrt(self.tau_sq), self.num_obv[task]) for task in range(self.num_tasks)]
        self.Y = np.concatenate(self.y, axis=0)
        self.F = self.func(self.mu)
        self.num_samples = num_samples
        self.delta_files = ['delta_'+str(task+1)+'.csv']
        self.mu_file = 'mu.csv'
        self.joint_file ='joint_sampling.csv'

    @staticmethod
    def func(delta):
        fx =  1 / (1 + np.exp(delta))
        return fx

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
        diff_prior = self.joint_prior(delta_p, mu, sigma_sq) - self.joint_prior(delta_c, mu, sigma_sq)
        alpha = min(1, np.exp(diff_likelihood + diff_prior))
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
        sigma_sq_c = self.sigma_delta_sq
        mu = self.mu
        delta_c = self.delta
        fx_c = self.fx
        file = open(self.joint_file, 'w')
        for sample in range(1, self.num_samples):
            # Propose delta_1 and delta_2
            delta_p = delta_c + np.random.normal(0, self.s_delta_sq, self.num_tasks)
            # Evaluate function f
            fx_p = self.func(delta_p)
            # get joint acceptance ratio value
            alpha = self.joint_acceptance_ratio(delta_c, delta_p, self.y, fx_c, fx_p, mu, tau_sq_c, sigma_sq_c)
            u = np.random.uniform(0, 1)
            if u < alpha:
                delta_c = delta_p
                fx_c = fx_p
            # Drawing tau_sq
            tau_a = np.sum(self.num_obv)/2 + tau_a_prior;
            tau_b = tau_b_prior
            tau_b = tau_b + np.sum(np.array([np.sum(np.square(self.y[task] - delta_c[task]))/2 for task in range(self.num_tasks)]))
            tau_sq_c = 1 / np.random.gamma(tau_a, tau_b)
            # Save weights
            weights = np.concatenate((delta_c, np.array([mu]))).reshape(1, self.num_tasks + 1)
            np.savetxt(file, weights, delimiter=',')
            print('weights: ', weights)
            # Draw mu
            mu = np.random.normal(delta_c.mean(), np.sqrt(sigma_sq_c/2))
            # Drawing sigma_sq
            sig_a = 2/2 + sig_a_prior
            sig_b = np.sum(np.square(delta_c - mu))/2 + sig_b_prior
            sigma_sq_c = 1 / np.random.gamma(sig_a, sig_b);

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
        ax = plt.subplot(311)
        plt.title('Weight Density plot')
        plt.hist(weights[burnin:, 0], bins=50, alpha=0.7, facecolor='sandybrown', edgecolor='r', label='source', density=True)
        plt.legend()
        ax = plt.subplot(312)
        plt.hist(weights[burnin:, 1], bins=50, alpha=0.7, facecolor='C0', edgecolor='b', label='target', density=True)
        plt.legend()
        ax = plt.subplot(313)
        plt.hist(weights[burnin:, 2], bins=50, alpha=0.7, facecolor='C8', edgecolor='g', label='mu', density=True)
        plt.legend()
        plt.xlabel('Parameter value')
        plt.ylabel('Density')
        plt.savefig('joint_weight.png')
        plt.clf()


if __name__ == '__main__':
    num_samples = 11000
    print("Initializing... ")
    experiment = Experiment(num_samples)
    # print("Initialized! Now running sampler..")
    # for task in range(1, 4):
    #     experiment.task_sampler(task)
    print('Starting joint smapling...')
    experiment.joint_sampler()
    experiment.plot_histogram()
