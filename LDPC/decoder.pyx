import cython

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, exp, log, pow, abs

def is_codeword(H, c):
    '''
    Check if c is a codework generated from code corresponding to H

    Parameters
    ----------
    H : scipy.sparse.csr_matrix or np.ndarray
        Parity check matrix of given code (CSR recommended)
    c : np.ndarray
        Array of bits of codeword

    Returns
    -------
    bool
        True if codework belongs to H code, False otherwise

    '''
    parity_check_bits = H.dot(c) % 2
    return np.all(parity_check_bits == 0)

from time import time

import numpy as np
import scipy.sparse as sp

import LDPC
import specs

H = specs.get_expanded_H_matrix(2304, '1/2')
H = LDPC.SPMatrix(H)

n = H.shape[1]

def decoder(H, sigma_w, u_distrib=None, max_iterations=10):
    '''
    Create function to decode parity check code desctribed by H

    Parameters
    ----------
    H : LDPC.SPMatrix
        Parity check matrix of code

    sigma_w : float
        Noise variance (true or estimated)

    u_distrib : np.ndarray, optional
        Array of length k where i-th element contains P[u_i == 0]
        Default value is uniform distribution for all bits

    max_iterations : int, optional
        Number of iterations of message passing algorithm
        Default value is 10

    Returns
    -------
    callable
        Function that receives a real numbered channel measures
        array and that returns transmitted message estimate
    '''
    # code information
    n = H.shape[1]
    k = n - H.shape[0]

    # set uniform if input a priori distribution
    # is not given
    if u_distrib == None:
        u_distrib = np.empty((k,))
        u_distrib.fill(0.5)

    # fill channel LLR vector with a priori information
    message_a_priori_LLR = np.log(u_distrib) - np.log(1 - u_distrib)

    # a priori LLR is 0 for the parity check bits
    a_prori_LLR = np.concatenate((message_a_priori_LLR, np.zeros(n-k)))

    # create matrices to store backward and forward messages
    F = H.copy()
    B = H.copy()

    ### INIT
    # set all backward messages as constants (TODO check if 0 is good)
    B.update_all(lambda x: x*0)

    def decode(r):
        ### A PRIORI -> initialize a priori channel knowledge
        ch = -2 * r / sigma_w**2 + a_prori_LLR

        for _ in range(max_iterations):
            # compute Fij using this precomputed vector b
            b = B.apply_on_rows(sum)

            ### ESTIMATION -> marginalization

            # compute codeword estimates, using b
            c = b + ch
            c[cw >= 0] = 0
            c[cw < 0] = 1

            # if word is valid, return message estimate and exit
            if LDPC.is_codeword(H, c):
                return c[0:k]

            ### FORWARD -> variable nodes update

            for (i, j), _ in H.items():
                F[i, j] = b[i] - B[i, j] + ch[j]

            ### BACKWARD -> check nodes update

            # compute total row sign
            row_signs = F.apply_on_rows(LDPC.global_sign)

            # transform all values in F, which is valid
            # since they will be rewritten in next cycle
            F.update_all(LDPC.phi_tilde_vector)

            # precompute total phi value on F rows
            row_phi = F.apply_on_rows(sum)

            # compute sum of phi values in the whole row
            for (i, j), _ in H.items():
                # check if global LLR sign is positive or negative
                if F[i, j] < 0:
                    current_row_sign = - row_signs[i]
                else:
                    current_row_sign = row_signs[i]

                # note that F[i, j] contains precoputed phi functions
                B[i, j] = current_row_sign * \
                          LDPC.phi_tilde(row_phi[i] - F[i, j])

        # valid codeword was not found up to max_iterations so declare
        # failure, returning a vector that, by NaN specification, fails
        # all equality tests with common numbers
        u = np.empty( (k,) )
        u.fill(np.nan)
        return u

    return decode

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef phi_tilde_vector(np.ndarray[np.double_t, ndim=1] x):
    '''
    Compute function phi tilde

    Parameters
    ----------
    x : np.ndarray[np.double_t, ndim=1]
        Unidimensional double array

    Returns
    -------
    np.ndarray[np.double_t, ndim=1]
        Array with same shape of x where y[i] = phi(x[i])
    '''
    y = np.empty(x.shape[0])
    for i, value in enumerate(x):
        y[i] = phi_tilde(value)
    return y

cpdef phi_tilde(double x):
    '''
    Compute function phi tilde

    Parameters
    ----------
    x : float
        Indipendent real variable

    Returns
    -------
    float
        Phi tilde computed in x
    '''
    x = abs(x)
    if x < 1e-5:
        return 12
    elif x > 12:
        return 0

    # compute function exactly
    k = exp(-x)
    return log( (1+k)/(1-k) )

cpdef global_sign(x):
    '''
    Compute sign of product of element in x
    Note that 0 is considered positive here.

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    int
        +1 if global sign is positive
        -1 otherwise
    '''
    if np.count_nonzero(x < 0) % 2 == 0:
        return 1
    return -1
