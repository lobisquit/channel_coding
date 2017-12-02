from math import floor
from pathlib import Path

import pandas as pd

SPECS_PATH = Path('specs')

def get_compressed_H_matrices():
    '''
    Get compressed H matrices for each code rate

    Returns
    -------
    dict {str, np.array}
        Compressed encoding matrix for each rate, defined by a string
    '''
    return {parse_rate(f): pd.read_csv(f, header=None)
            for f in SPECS_PATH.glob('H-*')}

def get_expanded_H_matrices(n, rate):
    '''
    Get expanded H matrices for each code rate

    Parameters
    ----------
    n: integer
        Code length
    rate: str
        String specifying code rate (ex. '2/3A')

    Returns
    -------
    dict {str, np.array}
        Expanded encoding matrix for each rate, defined by a string
    '''
    return {rate: expand_H(matrix, n, rate)
            for rate, matrix in get_compressed_H_matrices().items()}

def parse_rate(path):
    '''
    Get code rate from filename and put a bar
    between numerator and denominator
    '''
    rate = path.stem.split('-')[-1]
    return rate[:1] + '/' + rate[1:]

def get_block_sizes():
    ''' Get block sizes table for each code rate '''
    return {parse_rate(f): pd.read_csv(f)
            for f in SPECS_PATH.glob('block-size-*')}

def expander(num, n, rate):
    '''
    Expand single element of "compressed" matrix, given code
    length and code rate

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
    np.array
        Matrix expansion of single element
    '''
    zf = n // 24

    if zf < 1:
        raise ValueError(
            "Code length n invalid: {n}<24".format(n=n))

    if num < 0:
        return np.zeros( (zf, zf) )
    else:
        # compute expansion factor
        zf = n // 24

        # treat special rate 2/3A differently, as specified
        if rate == '2/3A':
            p = num % zf
        else:
            z0 = 96
            p = floor(num * zf / z0)

        # return a circular right shift of identity matrix, of extent p
        return np.roll(np.eye(zf), p)

def expand_H(matrix, n, rate):
    '''
    Apply expander function to each element on matrix,
    then merge all results in a bigger matrix

    Parameters
    -----------
    matrix: np.array or pd.DataFrame
        Any numpy array or DataFrame of integer numbers
    n: integer
        Code length
    rate: str
        String specifying code rate (ex. '2/3A')

    Returns
    -------
    np.array
        Expanded matrix, given code length and rate
    '''

    # convert dataframe to bare matrix if needed
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values

    # collect all expansions of each line in a list
    out_lines = []
    for line in matrix:
        out_line = []
        for element in line:
            out_line.append(expander(element, n, rate))

        # stack expansions of a line
        out_lines.append(np.hstack(out_line))

    # stack all lines together
    return np.vstack(out_lines)
