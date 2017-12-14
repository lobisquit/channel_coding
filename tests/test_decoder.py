from math import exp, log

import numpy as np

import LDPC
import specs


def phi_definition(x):
    # those thresholds were proposed during the lesson
    if x < 1e-5:
        return 12
    if x > 12:
        return 0
    k = exp(-x)
    return log( (1+k) / (1-k) )

def test_phi():
    x = np.logspace(-7, 15)
    y = LDPC.phi_tilde(x)

    for i, x_value in enumerate(x):
        assert phi_definition(x_value) == y[i], i

def test_SPMatrix_todense():
    ### all-zero matrix
    H = np.zeros( (3, 3) )

    H_prime = LDPC.SPMatrix(H).todense()
    if not np.all(H_prime == H):
        raise Exception('Dense SPMatrix conversion from dense is not reversible')


    ### sparse matrix
    H = np.zeros( (3, 3) )
    H[1, 1] = 1

    H_prime = LDPC.SPMatrix(H).todense()
    if not np.all(H_prime == H):
        raise Exception('Sparse SPMatrix conversion is not reversible')

    ### dense matrix
    H = np.ones( (3, 3) )
    H[1, 1] = 0

    H_prime = LDPC.SPMatrix(H).todense()
    if not np.all(H_prime == H):
        raise Exception('Dense SPMatrix conversion is not reversible')

    ### LDPC biggest encoding matrix
    H = specs.get_expanded_H_matrix(2304, '1/2')

    H_prime = LDPC.SPMatrix(H).todense()
    if not np.all(H_prime == H):
        raise Exception('LDPC SPMatrix conversion is not reversible')

def test_SPMatrix_get_element():
    H = np.zeros( (3, 4) )
    H[1, 1] = 1
    M = LDPC.SPMatrix(H)

    ### nonzero element
    if not M.get(1, 1) == H[1, 1]:
        raise Exception('Get non-zero element failed')

    ### zero element
    if not M.get(0, 1) == 0:
        raise Exception('Get zero element failed')

    ### zero element before nonzero one
    # this is causes an early escape from column index loop
    if not M.get(1, 0) == 0:
        raise Exception('Get zero element exceeding last column member failed')
