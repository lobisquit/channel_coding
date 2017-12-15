import cython

import numpy as np
cimport numpy as np

from .primitives import *

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
        Defaults to uniform distribution for all bits

    max_iterations : int, optional
        Number of iterations of message passing algorithm

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
            c[c >= 0] = 0
            c[c < 0] = 1

            # if word is valid, return message estimate and exit
            if LDPC.is_codeword(H, c):
                return c[0:k]

            ### FORWARD -> variable nodes update

            for (i, j), _ in H.items():
                F[i, j] = b[i] - B[i, j] + ch[j]

            ### BACKWARD -> check nodes update

            # compute total row sign
            row_signs = F.apply_on_rows(global_sign)

            # transform all values in F, which is valid
            # since they will be rewritten in next cycle
            F.update_all(phi_tilde_vector)

            # precompute total phi value on F rows
            row_phis = F.apply_on_rows(sum)

            # compute sum of phi values in the whole row
            for (i, j), _ in H.items():
                # check if global LLR sign is positive or negative
                if F[i, j] < 0:
                    current_row_sign = - row_signs[i]
                else:
                    current_row_sign = row_signs[i]

                # note that F[i, j] contains precoputed phi functions
                B[i, j] = current_row_sign * \
                          phi_tilde(row_phis[i] - F[i, j])

        # valid codeword was not found up to max_iterations: declare then
        # failure, returning a vector that, by NaN specification, fails
        # all equality tests with common numbers
        u = np.empty( (k,) )
        u.fill(np.nan)
        return u

    return decode
