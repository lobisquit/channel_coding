import argparse
from math import sqrt
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import LDPC
import specs

for n in specs.get_code_lengths():
    for rate in specs.get_code_rates():
        H = specs.get_expanded_H_matrix(n, rate)
        col_sum = H.sum(axis=0)
        row_sum = H.sum(axis=1)

        regular_per_rows = False
        if np.all(row_sum == row_sum[0]):
            regular_per_rows = True
            row_extent = row_sum[0]

        regular_per_cols = False
        if np.all(col_sum == col_sum[0]):
            regular_per_cols = True
            col_extent = col_sum[0]

        if regular_per_cols and regular_per_rows:
            print('n={}, rate={}: both regular'.format(n, rate))
        elif regular_per_cols:
            print('n={}, rate={}: regular per cols of {}'.format(n, rate, col_extent))
        elif regular_per_rows:
            print('n={}, rate={}: regular per rows of {}'.format(n, rate, row_extent))
        else:
            # print('non regular')
            pass
