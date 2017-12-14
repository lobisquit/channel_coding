import numpy as np
import scipy.sparse as sp

from libc.math cimport sqrt, exp, log, pow

cimport cython
cimport numpy as np

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
    y = x.copy()

    for i, value in enumerate(x):
        # put a threshold to value too close to 0 or too high
        if value < 1e-5:
            y[i] = 12
        elif value > 12:
            y[i] = 0
        else:
            # compute exactly function
            k = exp(-value)
            y[i] = log( (1+k)/(1-k) )
    return y

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

cdef class SPMatrix:
    '''
    This custom class to describe a Sparse Matrix of
    double precision floating point numbers.

    Its structure is meant for fast access and value
    update, while adding new values is not allowed.
    '''

    cdef long[:] row_pointers
    cdef long[:] col_indices
    cdef double[:] data

    cdef int n_rows
    cdef int n_cols

    def __init__(self, matrix):
        '''
        Parameters
        ----------
        matrix : np.ndarray of numbers
            Dense matrix to copy in sparse representation
        '''
        self.n_rows = matrix.shape[0]
        self.n_cols = matrix.shape[1]

        row_pointers = []
        col_indices = []
        data = []

        # first line starts at index 0 by default
        row_pointers.append(0)
        for i in range(0, self.n_rows):
            for j in range(0, self.n_cols):
                value = matrix[i, j]
                if value != 0:
                    data.append(<double> value)
                    col_indices.append(j)

            # add pointer to the next element to
            # mark the end of the current line
            row_pointers.append(len(data))

        self.row_pointers = np.array(row_pointers, dtype=np.long)
        self.col_indices = np.array(col_indices, dtype=np.long)
        self.data = np.array(data, dtype=np.double)

    def get(self, int i, int j):
        '''
        Retrieve element

        Parameters
        ----------
        i : int
            Row index
        j : int
            Column index

        Returns
        -------
        float
            Element (i, j) of matrix
        '''
        # get start and ending element of i-th row
        row_start = self.row_pointers[i]
        row_stop = self.row_pointers[i + 1]

        cdef int el_index
        # find element with matching column index
        for el_index in range(row_start, row_stop):
            col_index = self.col_indices[el_index]
            if col_index >= j:
                if col_index == j:
                    return self.data[el_index]
                else:
                    # col_index > j implies that element is empty
                    # otherwise it would have already been reached
                    return 0
        return 0

    def todense(self):
        '''
        Revert to dense representation

        Returns
        -------
        np.ndarray
            Dense representation
        '''
        matrix = np.zeros( (self.n_rows, self.n_cols) )

        cdef int i, el_index
        for i in range(self.n_rows):
            start = self.row_pointers[i]
            stop = self.row_pointers[i + 1]

            # loop through element of current line
            for el_index in range(start, stop):
                j = self.col_indices[el_index]
                value = self.data[el_index]

                matrix[i, j] = value
        return matrix

    def update(self, func):
        '''
        Update all data element by element and so
        regardless of their positions

        Parameters
        ----------
        func : callable
            Function to apply on a array returning
            a new array of same shape and type
        '''
        self.data = func(np.asarray(self.data))

    def along_rows(self, func):
        '''
        Operate the same function on all non-zero
        elements of each row

        Parameters
        ----------
        func : callable
            Function to apply on an array returning a float
        Returns
        -------
        np.ndarray
            Scores of each row
        '''
        # create output vector to store each line
        # result of func
        out = np.zeros(self.n_rows)

        cdef int i
        for i in range(self.n_rows):
            start = self.row_pointers[i]
            stop = self.row_pointers[i + 1]

            # apply func on current line values
            out[i] = func(np.asarray(self.data[start:stop]))
        return out
