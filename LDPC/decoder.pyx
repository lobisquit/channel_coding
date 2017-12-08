import numpy as np

from libc.math cimport sqrt, log, exp, pow

cimport cython
cimport numpy as np

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def phi_tilde(np.ndarray[DTYPE_t, ndim=1] x):
    '''
    Apply function phi tilde to an array, modifying it in place

    Parameters
    ----------
    x : np.ndarray[np.double, ndim=1]
        Unidimensional double array
    '''
    for i, value in enumerate(x):
        # put a threshold to value too close to 0 or too high
        if value < 1e-5:
            x[i] = 12
        elif value > 12:
            x[i] = 0
        else:
            # compute exactly function
            k = exp(-value)
            x[i] = log( (1+k)/(1-k) )

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
