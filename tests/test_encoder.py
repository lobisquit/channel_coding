import random

import numpy as np
import numpy.linalg as npla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import LDPC
from specs import *


def test_encoding_np():
    ''' Test encoding using dense matrices '''
    for n in get_code_lengths():
        for rate in get_code_rates():
            H = get_expanded_H_matrix(n, rate) # (n-k) x n

            n = H.shape[1]
            k = n - H.shape[0]

            B = H[0:(n-k):1, 0:k:1] # (n-k) x k
            C = H[0:(n-k):1, k:n:1] # (n-k) x (n-k)

            A = npla.inv(C).dot(B)

            G = np.vstack((np.eye(k), A))

            for trial in range(0, 10):
                i = random.randrange(0, 2**k)
                u = np.array([int(a) for a in np.binary_repr(i, width=k)])

                c = G.dot(u)
                if sum(H.dot(c) % 2) != 0:
                    raise ValueError("Error for n={}, rate={}, trial={}".format(n, rate, trial))

def test_encoding_lib():
    ''' Test encoding using sparse matrices of LDPC module '''
    for n in get_code_lengths():
        for rate in get_code_rates():
            H = get_expanded_H_matrix(n, rate) # (n-k) x n
            n = H.shape[1]
            k = n - H.shape[0]

            encoder = LDPC.encoder(H)

            for trial in range(0, 10):
                i = random.randrange(0, 2**k)
                u = np.array([int(a) for a in np.binary_repr(i, width=k)])

                c = encoder(u)
                if sum(H.dot(c) % 2) != 0:
                    raise ValueError("Error for n={}, rate={}, trial={}".format(n, rate, trial))
