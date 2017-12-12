import numpy as np
import scipy.sparse as sp
from libc.math cimport sqrt, log, exp, pow

cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def phi_tilde(np.ndarray[np.double_t, ndim=1] x):
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

cdef class Matrix:
    cdef long[:] row_pointers
    cdef long[:] col_indices
    cdef double[:] data

    cdef int n_rows
    cdef int n_cols

    def __init__(self, matrix):
        self.n_rows = matrix.shape[0]
        self.n_cols = matrix.shape[1]

        elements = matrix.nonzero()
        rows, cols = elements[0], elements[1]

        # convert rows vector of indices in vector of pointers in
        # column and data vectors
        row_pointers = []

        cdef int index
        for index in range(len(rows)):
            if index == 0:
                row_pointers.append(index)

            elif rows[index] != rows[index - 1]:
                for _ in range(rows[index-1], rows[index]):
                    row_pointers.append(index)

        # loop through ending lines, if empty
        for _ in range(rows[-1], self.n_rows - 1):
            row_pointers.append(self.n_rows)

        # store everything in object
        self.row_pointers = np.array(row_pointers, dtype=long)
        self.col_indices = np.array(cols, dtype=long)
        self.data = np.array(matrix.data, dtype=np.double)

    def tocsr(self):
        # create numpy array from data
        cols = np.frombuffer(self.col_indices, dtype=long)
        data = np.frombuffer(self.data, dtype=np.double)

        # rebuild rows array
        rows = []

        cdef int index
        for index in range(len(self.row_pointers)):
            start = self.row_pointers[index]
            if index == len(self.row_pointers) - 1:
                stop = self.n_rows
            else:
                stop = self.row_pointers[index + 1]

            for _ in range(start, stop):
                rows.append(index)

        print(rows)
        print(data)
        print(cols)

        coo_matrix = sp.coo_matrix((data, (rows, cols)),
                                   shape=(self.n_rows, self.n_cols))
        return coo_matrix.tocsr()
