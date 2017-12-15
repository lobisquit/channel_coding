from math import sqrt
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import LDPC
import specs

## main setup
Eb = 1/2
SNRs = np.arange(0.5, 4.6, step=0.5) ## Eb/N0

results = pd.DataFrame()
for n in specs.get_code_lengths():
    for rate in specs.get_code_rates():
        H = specs.get_expanded_H_matrix(n, rate)
        k = n - H.shape[0]

        # setup encoder functions
        enc = LDPC.encoder(H)

        for SNR in SNRs:
            sigma_w = sqrt(Eb / SNR)

            # setup decoder functions
            dec = LDPC.decoder(H, sigma_w)

            # count number of tested words, errors and
            # number of iterations needed for convergence
            n_words = 0
            n_errors = 0
            total_n_iterations = 0

            while n_errors <= 10 and n_words < 100:
                u = np.random.choice(a=[0, 1], size=k)

                c = enc(u)                     ## ENCODE
                d = LDPC.modulate(c)           ## MODULATE
                r = LDPC.channel(d, sigma_w)   ## add CHANNEL noise

                u_prime, n_iterations = dec(r) ## DECODE
                total_n_iterations += n_iterations

                if not np.all(u == u_prime):
                    n_errors += 1
                n_words += 1

                print('SNR = {}, word = {}, n_iter = {}' \
                      .format(SNR, n_words, n_iterations), end='\r')

            current_result = pd.DataFrame({
                'n': [n],
                'rate': [rate],
                'SNR' : [SNR],
                'Pe' : n_errors / n_words,
                'n_iterations' : total_n_iterations / n_words,
            })

            results = pd.concat([results, current_result])
        break
    break

print(results)
