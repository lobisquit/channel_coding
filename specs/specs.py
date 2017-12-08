# -*- coding: utf-8 -*-
import warnings
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

SPECS_PATH = Path('specs')
''' pathlib.Path
        Path of current file subfolder
'''

# ignore warnings related to CSR sparse matrix format
# since here we do not solve linear systems
warnings.filterwarnings("ignore", message="splu requires CSC matrix format")
warnings.filterwarnings("ignore", message="spsolve is more efficient when sparse b is in the CSC matrix format")


def get_code_rates():
    return [parse_rate(rate) for rate in SPECS_PATH.glob('H-*')]

def get_code_lengths():
    '''

    Returns
    -------
    list
        Supported code lengths n in bits
    '''
    # note that rate here is chosen arbitrarily, since
    # all rates support all code lengths
    block_size_table = get_block_size(get_code_rates()[0])
    return list(block_size_table['n (bits)'])

def get_compressed_H_matrix(rate):
    '''
    Get compressed H matrices for given code rate

    Parameters
    ----------
    rate : str
        Code rate label (ex. '2/3A')

    Returns
    -------
    np.ndarray
        Compressed encoding matrix
    '''
    files = SPECS_PATH.glob('H-*')
    for f in files:
        if parse_rate(f) == rate:
            return pd.read_csv(f, header=None).values

    raise ValueError('Invalid rate: {}'.format(rate))

def expander(num, n, rate):
    '''
    Expand single element of "compressed" matrix, given code
    length and code rate.
    See p(zf, i, j) in specifications.

    Parameters
    ----------
    num: integer
        Element of "compressed" matrix
    n: integer
        Code length
    rate: str
        String specifying code rate (ex. '2/3A')

    Returns
    -------
    np.ndarray
        Matrix expansion of single element, of shap (zf x zf)
    '''

    # compute expansion factor
    zf = n // 24
    if zf < 1:
        raise ValueError(
            "Code length n invalid: {n}<24".format(n=n))

    if num < 0:
        return np.zeros((zf, zf), dtype=int)
    else:
        # treat special rate 2/3A differently, as specified
        if rate == '2/3A':
            p = num % zf
        else:
            z0 = 96
            p = floor(num * zf / z0)

        # return a circular right shift of identity matrix, of extent p
        return np.roll(np.eye(zf, dtype=int), p, axis=1)

def get_expanded_H_matrix(n, rate):
    '''
    Get expanded H matrices for given code rate

    Parameters
    ----------
    n : int
        Code length
    rate : str
        Code rate label (ex. '2/3A')

    Returns
    -------
    sp.csr_matrix
        Compressed encoding matrix, optimized for matrix-vector product
    '''
    # collect all expansions of each line in a list
    out_lines = []
    for line in get_compressed_H_matrix(rate):
        out_line = []
        for element in line:
            out_line.append(expander(element, n, rate))

        # stack expansions of a line
        out_lines.append(np.hstack(out_line))

    # stack all lines together
    H = np.vstack(out_lines)
    return sp.csr_matrix(H)

def parse_rate(path):
    '''
    Get code rate from filename and put a bar
    between numerator and denominator

    Parameters
    ----------
    path : pathlib.Path
        File of specific code rate

    Returns
    -------
    str
        Code rate label
    '''
    rate = path.stem.split('-')[-1]
    return rate[:1] + '/' + rate[1:]

def get_block_size(rate):
    '''
    Get block sizes table for a code rate

    Parameters
    ----------
    rate : str
        Code rate label (ex. '2/3A')

    Returns
    -------
    pd.DataFrame
        Code specifications table for given rate
    '''
    files = SPECS_PATH.glob('block-size-*')
    for f in files:
        if parse_rate(f) == rate:
            return pd.read_csv(f)

    raise ValueError('Invalid rate: {}'.format(rate))
