import cython
import numpy as np

from libc.math cimport sqrt, abs, exp, log, pow

cimport numpy as np

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
        return 12.206 # phi_tilde(1e-5)
    elif x > 12:
        return 0

    # compute function exactly
    k = exp(-x)
    return log( (1+k)/(1-k) )

cpdef global_sign(x):
    '''
    Compute sign of product of non-zero elements in x
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
