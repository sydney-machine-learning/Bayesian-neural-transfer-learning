# Bayesian-neural-transfer-learning
A transfer learning scheme for Bayesian Neural Networks.

![bntl](https://github.com/sydney-machine-learning/Bayesian-neural-transfer-learning/blob/master/BNTL.png)

## How to run Bayesian neural transfer learning

1. Create the experiment directory

    ```bash
    $ mkdir mcmc-transfer-learning/exp
    ```

2. Generate Synthetic Data (Optional)

    ```bash
    $ cd datasets
    $ python3 generate_synthetic_data.py
    ```

    *Note: this step is only required if you plan to run BNTL for Synthetic Dataset.*

3. Run the experiment

    ```bash
    $ cd mcmc-transfer-learning/scripts/bntl
    $ python3 ldbntl.py --langevin --problem 3 --langevin-ratio 0.1 --run-id 1 --root-dir /home/arpitk/projects/Bayesian-neural-transfer-learning
    ```

    *Note: Replace the root-dir path (/home/arpitk/projects/Bayesian-neural-transfer-learning) with the path to Bayesian-neural-transfer-learning repository on your machine* 

    More details on the available arguments:
    ```
    $ python3 ldbntl.py --help

    usage: ldbntl.py [-h] [--langevin] --problem PROBLEM [--num-samples NUM_SAMPLES] [--transfer-only] [--no-transfer] [--langevin-ratio LANGEVIN_RATIO] [--transfer-ratio TRANSFER_RATIO]
                    --run-id RUN_ID [--root-dir ROOT_DIR]

    Run Bayesian neural transfer learning

    optional arguments:
    -h, --help            show this help message and exit
    --langevin            use langevin gradients
    --problem PROBLEM     constant value defining the problem to use: 0:- Wine-Quality, 1: UJIndoorLoc, 2: Sarcos, 3: Synthetic
    --num-samples NUM_SAMPLES
                            total number of samples sampled by the mcmc sampler
    --transfer-only       don't sample target without transfer (default=False)
    --no-transfer         no transfer
    --langevin-ratio LANGEVIN_RATIO
                            the ratio of samples to use langevin gradients
    --transfer-ratio TRANSFER_RATIO
                            the ratio of samples to be transfered
    --run-id RUN_ID
    --root-dir ROOT_DIR   Path to root directory of repository

    ```