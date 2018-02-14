# -*- coding: utf-8 -*-
import cython
import numpy as np

cimport numpy as np

def modulate(np.ndarray[long, ndim=1] u):
    '''
    Map input message symbols to modulation ones
    modulate : x
      | 0 -> -1.0
      | 1 -> +1.0

    Parameters
    ----------
    c : np.ndarray[np.double_t, ndim=1]
        Codeword

    Returns
    -------
    np.ndarray
        Transmitted signal
    '''
    return (u * 2 - 1).astype(np.double)

def channel(np.ndarray[np.double_t, ndim=1] d, double sigma_w):
    '''
    Add noise to modulation symbols d

    Parameters
    ----------
    d : np.ndarray[np.double_t, ndim=1]
        Modulation word
    sigma_w : float
        Square root of noise variance

    Returns
    -------
    np.ndarray
        Received signal
    '''
    w = sigma_w * np.random.randn(len(d))
    return d + w
