# -*- coding: utf-8 -*-
import cython
import numpy as np
import scipy.sparse as sp

from .primitives import *
from .structures import *

cimport numpy as np

def codeword_checker(H):
    '''
    Check if c is a codework generated from code corresponding to H

    Parameters
    ----------
    H : scipy.sparse.csr_matrix or np.ndarray
        Parity check matrix of given code (CSR recommended)

    Returns
    -------
    callable
        Function that returns True if codework belongs to H code
        and False otherwise
    '''
    sparse_H = sp.csr_matrix(H)

    def is_codeword(c):
        parity_check_bits = sparse_H.dot(c) % 2
        return np.all(parity_check_bits == 0)

    return is_codeword

def decoder(H, sigma_w, u_distrib=None, max_iterations=10):
    '''
    Create function to decode parity check code desctribed by H

    Parameters
    ----------
    H : np.ndarray
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
        array and that returns transmitted message estimate and
        number of iterations of message passing performed
    '''
    # create function to check words of given code
    is_codeword = codeword_checker(H)

    # convert to sparse representation
    H = SPMatrix(H)

    # code information
    n = H.shape[1]
    k = n - H.shape[0]

    # set uniform if input a priori distribution is not given
    if u_distrib == None:
        u_distrib = np.empty( (k,) )
        u_distrib.fill(0.5)

    # fill channel LLR vector with a priori information
    message_a_priori_LLR = np.log(u_distrib) - np.log(1 - u_distrib)

    # a priori LLR is 0 for the parity check bits
    a_prori_LLR = np.concatenate((message_a_priori_LLR, np.zeros(n-k)))

    # create matrices to store backward and forward messages
    F = H.copy()
    B = H.copy()

    def decode(r):
        ### INIT
        nonlocal B, F # explicity use higher lever matrices

        # set all backward messages as constants
        B = B.update_all(lambda x: x * 0)

        ### A PRIORI -> initialize a priori channel knowledge
        ch = -2 * r / sigma_w**2 + a_prori_LLR

        cdef int current_iter
        for current_iter in range(max_iterations):
            # precompute vector b, sum of columns of B
            b = B.sum_on_columns()

            ### ESTIMATION -> marginalization

            # compute codeword estimates, using b
            c = b + ch
            c[c >= 0] = 0
            c[c < 0] = 1

            # if word is valid, return message estimate and exit
            if is_codeword(c):
                return c[0:k], current_iter

            ### FORWARD -> variable nodes update

            for (i, j), _ in H.items():
                # term between brackets is the Backward message sum
                # across all j-th column but the i-th element
                F[i, j] = (b[j] - B[i, j]) + ch[j]

            ### BACKWARD -> check nodes update

            # compute total row sign
            row_signs = F.apply_on_rows(global_sign)

            # transform all values in F, which is valid
            # since they will be rewritten in next cycle
            PHI = F.update_all(phi_tilde_vector)

            # precompute sum of phi value of each row
            row_phi = PHI.apply_on_rows(sum)

            # compute sum of phi values in the whole row
            for (i, j), _ in H.items():
                # check if global LLR sign is positive or negative
                if F[i, j] < 0:
                    current_row_sign = - row_signs[i]
                else:
                    current_row_sign = row_signs[i]

                # phi tilde is computed on all terms in i-th row
                # but the j-th element
                B[i, j] = current_row_sign * \
                          phi_tilde(row_phi[i] - PHI[i, j])

        # valid codeword was not found up to max_iterations so declare
        # failure, returning a vector that, by NaN specification, fails
        # all equality tests
        u = np.empty( (k,) )
        u.fill(np.nan)
        return u, max_iterations

    return decode
