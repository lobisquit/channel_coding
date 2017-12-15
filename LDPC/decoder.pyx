import numpy as np
import scipy.sparse as sp

from libc.math cimport sqrt, exp, log, pow

cimport cython
cimport numpy as np

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

@cython.boundscheck(False)
@cython.wraparound(False)
def phi_tilde(np.ndarray[np.double_t, ndim=1] x):
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
