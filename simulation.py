from math import sqrt
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import LDPC
import specs

## simulation parameters

# maximum number of errors and words per (n, rate, SNR) configuration
MAX_N_ERRORS = 10
MAX_N_WORDS = 100

# maximum number of iterations of message passing algorithm
# NOTE that time per word is roughly upper bounded by 0.8 * MAX_ITERATIONS
# (0.8s / cycle was assessed for n=2304, rate 1/2, biggest matrix)
MAX_ITERATIONS = 10

## main setup
SNRs = np.arange(0.5, 4.6, step=0.5) ## Eb/N0

results = []
for n in specs.get_code_lengths():
    for rate in specs.get_code_rates():
        H = specs.get_expanded_H_matrix(n, rate)
        k = n - H.shape[0]

        # setup encoder functions
        enc = LDPC.encoder(H)

        for SNR in SNRs:
            print('n = {}, rate = {}, SNR = {}'.format(n, rate, SNR))

            # compute noise standard deviation, given a binary code (M=2)
            # of rate ?? used in a passband communication
            sigma_w = sqrt(1 / SNR)

            # setup decoder functions
            dec = LDPC.decoder(H, sigma_w, max_iterations=MAX_ITERATIONS)

            # count number of tested words, errors, failures and
            # number of iterations needed for convergence
            n_words = 0
            n_errors = 0
            n_failures = 0
            n_iterations = 0

            # measure total time taken per word
            start = time()
            while n_errors + n_failures < MAX_N_ERRORS and n_words < MAX_N_WORDS:
                print('n_errors = {}, n_failures = {}, n_words = {}'\
                      .format(n_errors, n_failures, n_words), end='\r')
                u = np.random.choice(a=[0, 1], size=k)

                c = enc(u)                       ## ENCODE
                d = LDPC.modulate(c)             ## MODULATE
                r = LDPC.channel(d, sigma_w)     ## add CHANNEL noise

                u_prime, current_n_iter = dec(r) ## DECODE

                ## update PERFORMANCE measures

                n_iterations += current_n_iter
                n_words += 1

                # report failure if word decoding fails
                if np.all(np.isnan(u_prime)):
                    n_failures += 1
                # if decoded word was the wrong one, report an error
                elif not np.all(u_prime == u):
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
        break
    break

summary = pd.concat(results)
print(summary)
