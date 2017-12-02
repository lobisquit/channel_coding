from math import floor
from pathlib import Path

import pandas as pd

SPECS_PATH = Path('specs')

H_PATHS = list(SPECS_PATH.glob('H-*'))
BLOCK_SIZE_PATHS = list(SPECS_PATH.glob('block-size-*'))

def parse_rate(path):
    ''' Get code rate from filename and put a bar between the numbers '''
    rate = path.stem.split('-')[-1]
    return rate[:1] + '/' + rate[1:]

def get_H_matrices():
    ''' Get raw H matrices for each code rate '''
    return {parse_rate(f): pd.read_csv(f, header=None) for f in H_PATHS}

def get_block_sizes():
    ''' Get block sizes table for each code rate '''
    return {parse_rate(f): pd.read_csv(f) for f in BLOCK_SIZE_PATHS}

def expand_H(matrix, expander):
    '''
    Apply expander function to each element on matrix,
    then merge all results in matrix form

    Parameters
    -----------
    matrix: np.array
        Any numpy array
    expander: function
        Function that maps a number to np.array
    '''

    # collect all expansions of each line in a list
    out_lines = []
    for line in matrix:
        out_line = []
        for element in line:
            out_line.append(expander(element))

        # stack expansions of a line
        out_lines.append(np.hstack(out_line))

    # stack all lines together
    return np.vstack(out_lines)
