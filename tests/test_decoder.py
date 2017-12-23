from math import exp, log

import numpy as np
import pytest

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
    for value in np.logspace(-7, 2):
        assert phi_definition(value) == LDPC.phi_tilde(value)

def sign_of_vector(x):
    assert np.all(x != 0), 'Invalid vector to check sign'

    if np.prod(x) > 0:
        return 1
    else:
        return -1

def test_sign():
    # check against dummy cases
    correct_ones = {
        (1, 2, 3) : +1,
        (1, -1, 3) : -1,
        (1) : +1,
        (-2, -2): +1,
    }

    for vector, correct_sign in correct_ones.items():
        assert LDPC.global_sign(np.array(vector)) == correct_sign

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

@pytest.mark.specs
def test_SPMatrix_todense_with_specs():
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
    if not M[1, 1] == H[1, 1]:
        raise Exception('Get non-zero element failed')

    ### zero element
    if not M[0, 1] == 0:
        raise Exception('Get zero element failed')

    ### zero element before nonzero one
    # this is causes an early escape from column index loop
    if not M[1, 0] == 0:
        raise Exception('Get zero element exceeding last column member failed')
