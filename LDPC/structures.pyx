# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np

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

    @property
    def shape(self):
        '''
        Get 2D matrix shape

        Returns
        -------
        (int, int)
            (number of rows, number of columns)
        '''
        return (self.n_rows, self.n_cols)

    def copy(self):
        '''
        Return a copy of current matrix

        Returns
        -------
        LDPC.SPMatrix
            Copy of self
        '''
        # create dummy SPMatrix object, to fill with self properties
        dummy_matrix = ( (0,), (0,) )
        matrix = SPMatrix(np.array(dummy_matrix))

        matrix.row_pointers = self.row_pointers.copy()
        matrix.col_indices = self.col_indices.copy()
        matrix.data = self.data.copy()

        matrix.n_rows = self.n_rows
        matrix.n_cols = self.n_cols

        return matrix

    def get_index_in_data(self, pos):
        cdef int i, j
        i, j = pos

        # get start and ending element of i-th row
        row_start = self.row_pointers[i]
        row_stop = self.row_pointers[i + 1]

        cdef int el_index
        # find element with matching column index
        for el_index in range(row_start, row_stop):
            col_index = self.col_indices[el_index]
            if col_index >= j:
                if col_index == j:
                    return el_index
                else:
                    # col_index > j implies that element is empty
                    # otherwise it would have already been reached
                    return -1
        return -1

    def __getitem__(self, pos):
        '''
        Retrieve element

        Parameters
        ----------
        pos : (int, int)
            (row, col) position to inspect

        Returns
        -------
        float
            element (i, j) of matrix, 0 if position is outside matrix
        '''
        cdef int el_index
        el_index = self.get_index_in_data(pos)

        if el_index == -1:
            return 0
        else:
            return self.data[el_index]

    def __setitem__(self, pos, value):
        '''
        Retrieve element

        Parameters
        ----------
        pos : (int, int)
            (row, col) position to inspect

        Returns
        -------
        float
            Element (i, j) of matrix, 0 if position is outside matrix
        '''
        cdef int el_index
        el_index = self.get_index_in_data(pos)

        if el_index == -1:
            raise ValueError('Position ({i}, {j}) is empty in SPMatrix \
                             unable to assign'.format(i=pos[0], j=pos[1]))
        else:
            self.data[el_index] = value

    def update_all(self, func):
        '''
        Update all data element by element and so
        regardless of their positions

        Parameters
        ----------
        func : callable
            Function to apply on a array returning
            a new array of same shape and type

        Returns
        -------
        LDPC.SPMatrix
            Copy of self with values updated
        '''
        cdef SPMatrix other = self.copy()
        other.data = func(np.asarray(other.data))

        return other

    def items(self):
        '''
        Retrieve all items of matrix, with their positions

        Returns
        -------
        generator
           Generator of ((i, j), value) tuples
        '''
        cdef int i, el_index
        for i in range(self.n_rows):
            start = self.row_pointers[i]
            stop = self.row_pointers[i + 1]

            # loop through element of current line
            for el_index in range(start, stop):
                j = self.col_indices[el_index]
                value = self.data[el_index]

                yield (i, j), value

    def apply_on_rows(self, func):
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

    def sum_on_columns(self):
        '''
        Sum elements over columns

        Returns
        -------
        np.ndarray
            Sum of elements in each column
        '''
        # create output vector to store each line
        # result of func
        out = np.zeros(self.n_cols)

        cdef int el_index, j
        for el_index, j in enumerate(self.col_indices):
            out[j] += self.data[el_index]

        return out
