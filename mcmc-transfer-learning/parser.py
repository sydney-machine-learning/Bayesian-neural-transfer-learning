import argparse

parser = argparse.ArgumentParser(description='Run Bayesian neural transfer learning')
parser.add_argument('--langevin', action='store_true', help='use langevin gradients')
parser.add_argument('--problem', type=int, default=0, help='constant value defining the problem to use: \n0:- Wine-Quality, 1: UJIndoorLoc, 2: Sarcos, 3: Synthetic', required=True)

parser.add_argument('--num-samples', type=int, required=False, help='total number of samples sampled by the mcmc sampler')

parser.add_argument('--transfer-only', action='store_true', help="don't sample target without transfer (default=False)")

args = parser.parse_args()
print(args)