import random

import numpy as np
import numpy.linalg as npla
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import LDPC
from specs import *


@pytest.mark.slowtest
@pytest.mark.specs
def test_generated_matrix():
    ''' Generated matrix columns must be in the null space of H '''
    for n in get_code_lengths():
        for rate in get_code_rates():
            H = sp.csc_matrix(get_expanded_H_matrix(n, rate))
            G = LDPC.get_generating_matrix(H)
            F = H.dot(G)

            # accessing data directly is the only way to apply modulo 2
            # since it is not natively supported by sp.csc_matrix
            F.data = F.data % 2
            if F.count_nonzero() != 0:
                raise Exception(
                    'Invalid generating matrix for n={}, rate={}'
                    .format(n, rate))

@pytest.mark.slowtest
@pytest.mark.specs
def test_encoder():
    for n in get_code_lengths():
        for rate in get_code_rates():
            H = sp.csc_matrix(get_expanded_H_matrix(n, rate))

            k = H.shape[1] - H.shape[0]
            u = np.random.choice(a=[0, 1], size=k, p=[1/2, 1/2])

            enc = LDPC.encoder(H)
            if np.count_nonzero(H.dot(enc(u)) % 2) != 0:
                raise Exception(
                    'Invalid encoding function for n={}, rate={}'
                    .format(n, rate))
