import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import inv


def get_generating_matrix(H):
    '''
    Convert parity check matrix H to its generating matrix G

    Parameters
    ----------
    H : scipy.sparse.csc_matrix or np.array
        Parity check matrix of given code

    Returns
    -------
    scipy.sparse.csc_matrix
        Sparse encoding matrix for given code
    '''

    if isinstance(H, np.ndarray):
        # sparsify matrix if needed
        H = sp.csc_matrix(H)

    # H has shape (n-k) x n
    n = H.shape[1]
    k = n - H.shape[0]

    B = H[0:(n - k):1, 0:k:1]        # (n-k) x k
    C = H[0:(n - k):1, k:n:1]        # (n-k) x (n-k)

    A = inv(C).dot(B)                # (n-k) x k
    G = sp.vstack((sp.eye(k), A))    # n x k

    return G

def encoder(H):
    '''
    Create encoder function from generating matrix

    Parameters
    ----------
    G : scipy.sparse.csc_matrix or np.array
        Generating matrix

    Returns
    -------
    callable
        Function that accepts vector u as input and
        returns encoded vector c (according to G)
    '''
    # here modulo is needed, since product could
    # lead to values different from 0 and 1
    G = get_generating_matrix(H)

    return lambda u: G.dot(u)
