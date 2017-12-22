import argparse
from math import sqrt
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

import LDPC
import specs

## simulation parameters

# maximum number of words per (n, rate, SNR) configuration
MAX_N_WORDS = 100
MIN_N_ERRORS = 100
MIN_N_CORRECT = 100

# maximum number of iterations of message passing algorithm
# NOTE that time per word is roughly upper bounded by 0.8 * MAX_ITERATIONS
# (0.8s / cycle was assessed for n=2304, rate 1/2, biggest matrix)
MAX_ITERATIONS = 10

## main setup
SNRs = np.arange(0.5, 4.6, step=0.5) ## Eb/N0

def step(n, rate):
    # extract rate from label (removing last letter if any)
    R = eval(rate[:3])

    H = specs.get_expanded_H_matrix(n, rate)
    k = n - H.shape[0]

    # setup encoder functions
    enc = LDPC.encoder(H)

    results = []
    for SNR in SNRs:
        # print('n = {}, rate = {}, SNR = {}'.format(n, rate, SNR))

        # compute noise standard deviation, given a binary PAM (M=2)
        # of rate R and SNR = Eb/N0
        # $ \sigma_w = \frac{E_s}{2R ` \log_2 M ` \Gamma} $
        # where $ E_s = 1, M=2, \Gamma = \frac{E_b}{N_0} $
        sigma_w = sqrt(1 / (2 * R * SNR))

        # setup decoder functions
        dec = LDPC.decoder(H, sigma_w, max_iterations=MAX_ITERATIONS)

        # count number of tested words, errors, failures and
        # number of iterations needed for convergence
        n_words = 0
        n_errors = 0
        n_failures = 0
        n_iterations = 0

        # generate always the same uniform messages,
        # in order to obtain smoother SNR-Pe curves
        np.random.seed(0)

        # measure total time taken per word
        start = time()

        # proceed until maximum word quota is exceeded or wanted
        # number of bad decoding and correct words is reached
        while n_words < MAX_N_WORDS and \
              (n_errors + n_failures < MIN_N_ERRORS or \
              n_words - (n_errors + n_failures) < MIN_N_CORRECT):
            # print('n_errors = {}, n_failures = {}, n_words = {}'\
                # .format(n_errors, n_failures, n_words), end='\r')

            u = np.random.choice(a=[0, 1], size=k)

            c = enc(u)                       ## ENCODE
            d = LDPC.modulate(c)             ## MODULATE
            r = LDPC.channel(d, sigma_w)     ## add CHANNEL noise

            u_prime, current_n_iter = dec(r) ## DECODE

            ## update PERFORMANCE measures

            n_iterations += current_n_iter
            n_words += 1

            if not np.all(u_prime == u):
                if np.all(np.isnan(u_prime)):
                    # report failure if word decoding fails
                    n_failures += 1
                else:
                    # if decoded word was the wrong one, report an error
                    n_errors += 1

        ## REPORT

        current_result = pd.DataFrame({
            'n'             : [n],
            'rate'          : rate,
            'SNR'           : SNR,
            # note that traditional wrong decoding probabilility
            # is the sum of the following ones, that are disjoint
            # "bad" decoding events
            'Perror'        : n_errors         / n_words,
            'Pfailure'      : n_failures       / n_words,
            'iterations'    : n_iterations     / n_words,
            'time per word' : (time() - start) / n_words,
            'n words'       : n_words
        })
        results.append(current_result)

    # collect results for current couple (n, rate)
    summary = pd.concat(results)
    summary.to_csv('results/SNRvsPe_n-{}_rate-{}.csv'\
                   .format(n, rate.replace('/', '')), index=None)

def configurations():
    for n in specs.get_code_lengths(): # [576, 672] # dummy
        for rate in specs.get_code_rates(): # ['1/2', '5/6'] # dummy
            yield n, rate

# read wanted number of processes
parser = argparse.ArgumentParser(description='Test LDPC codes.')
parser.add_argument('--processes', dest='processes', type=int, default=1)
cmd_args = parser.parse_args()

if cmd_args.processes == 1:
    for config in configurations():
        step(*config)
else:
    # perform all computations in parallel fashion if requested
    Parallel(jobs=cmd_args.processes)(delayed(step)(*config) for config in configurations())

# merge all resulting csv
csvs = Path('results/').glob('SNRvsPe_*.csv')
data = pd.concat([pd.read_csv(csv) for csv in csvs])
data.to_csv('results/SNRvsPe.csv', index=None)
