# !/usr/bin/python
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
        self.n1 = 20
        self.n2 = 80
        self.sigma_sq = 1
        self.tau_sq = 25
        self.s_sq = 0.25
        self.delta1, self.delta2 = np.random.normal(self.mu, self.sigma_sq, 2)
        self.y1 = np.random.normal(self.delta1, self.sigma_sq, self.n1)
        self.y2 = np.random.normal(self.delta2, self.sigma_sq, self.n2)
        self.y_mu = np.concatenate((self.y1, self.y2), axis=0)
        self.f1 = self.f(delta1)
        self.f2 = self.f(delta2)
        self.num_samples = num_samples

    @staticmethod
    def f(delta):
        delta = delta.astype(np.float128)
        fx =  1 / (1 + np.exp(delta))
        return fx

    @staticmethod
    def acceptance_ratio_delta(y, fx_c, fx_p):
        alpha = np.exp(-0.5 * np.sum(np.square(y - fx_p)) + 0.5 * np.sum(np.square(y - fx_c)))
        return alpha

    @staticmethod
    def acceptance_ratio_mu(y, f_c, f_p, mu_c, mu_p, tau_sq):
        alpha = np.exp(-0.5 * np.sum(np.square(y - f_p)) - 0.5 * mu_p/tau_sq) -  np.exp(-0.5 * np.sum(np.square(y - f_c)) -0.5 * mu_c/tau_sq)
        return alpha

    # task equals 1 or 2
    def sampler_delta(self, task):
        pass

    def smapler_joint(self):
        mu_c = self.mu
        f_mu_c = self.f(mu_c)
        delta1_c, delta2_c = self.delta1, self.delta2
        mu_samples = np.zeros(self.num_samples)
        mu_samples[0] = mu_c
        for sample in range(1, self.num_samples):
            # Propose mu
            mu_p = mu_c + np.random.normal(0, self.s_sq)
            f_mu_p = self.f(mu_p)
            # Calculate acceptance ratio for mu
            alpha_mu = self.acceptance_ratio_mu(self.y_mu, f_mu_c, f_mu_p, mu_c, mu_p, self.tau_sq)
            u = np.random.uniform(0,1)
            if u < alpha_mu:
                # Accept mu proposal
                mu_c = mu_p
                f_mu_c = f_mu_p
            mu_samples[sample] = mu_c

            # Propose delta1 and delta2
            
