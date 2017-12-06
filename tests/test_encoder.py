import random

import numpy as np
import numpy.linalg as npla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import LDPC
from specs import *


def test_encoding():
    ''' Test encoding using sparse matrices of LDPC module '''

    # test a subset of code lengths, in order to
    # keep testing fast enough
    for n in random.sample(get_code_lengths(), 5):
        for rate in get_code_rates():
            H = get_expanded_H_matrix(n, rate) # (n-k) x n
            n = H.shape[1]
            k = n - H.shape[0]

            encoder = LDPC.encoder(H)

            for trial in range(0, 50):
                u = np.random.choice(a=[0, 1], size=k, p=[1/2, 1/2])

                c = encoder(u)
                if sum(H.dot(c) % 2) != 0:
                    raise ValueError(
                        "Error for n={}, rate={}, trial={}"
                        .format(n, rate, trial)
                    )
