import cython
import numpy as np
import scipy.sparse as sp

from .primitives import *
from .structures import *

cimport numpy as np

np.set_printoptions(suppress=True, precision=3)

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

    ### INIT

    # set all backward messages as constants (TODO check if 0 is good)
    B = B.update_all(lambda x: x * 0)

    def decode(r):
        ### A PRIORI -> initialize a priori channel knowledge
        ch = -2 * r / sigma_w**2 + a_prori_LLR

        cdef int current_iter
        for current_iter in range(max_iterations):
            # print('------------------> {}'.format(current_iter))

            # precompute vector b, sum of columns of B
            b = B.sum_on_columns()

            # print('b')
            # print(b)
            # print('')

            ### ESTIMATION -> marginalization

            # compute codeword estimates, using b
            c = b + ch

            # print('c = {}'.format(c))
            c[c >= 0] = 0
            c[c < 0] = 1

            # if word is valid, return message estimate and exit
            if is_codeword(c):
                return c[0:k], current_iter

            ### FORWARD -> variable nodes update

            for (i, j), _ in H.items():
                F[i, j] = b[i] - B[i, j] + ch[j]

            # print('ch = {}\n'.format(ch))
            # print('F')
            # print(F.todense())
            # print('')

            ### BACKWARD -> check nodes update

            # compute total row sign
            row_signs = F.apply_on_rows(global_sign)

            # print('row_signs')
            # print(row_signs)
            # print('')

            # transform all values in F, which is valid
            # since they will be rewritten in next cycle
            PHI = F.update_all(phi_tilde_vector)

            # print('PHI')
            # print(PHI.todense())
            # print('')

            # precompute sum of phi value of each row
            row_phi = PHI.apply_on_rows(sum)
            # print('row_phi')
            # print(row_phi)
            # print('')

            # compute sum of phi values in the whole row
            for (i, j), _ in H.items():
                # check if global LLR sign is positive or negative
                if F[i, j] < 0:
                    current_row_sign = - row_signs[i]
                else:
                    current_row_sign = row_signs[i]

                # note that F[i, j] contains precoputed phi functions
                B[i, j] = current_row_sign * \
                          phi_tilde(row_phi[i] - PHI[i, j])

            # print('B')
            # print(B.todense())
            # print('')

        # valid codeword was not found up to max_iterations so declare
        # failure, returning a vector that, by NaN specification, fails
        # all equality tests with common numbers
        u = np.empty( (k,) )
        u.fill(np.nan)
        return u, max_iterations

    return decode
