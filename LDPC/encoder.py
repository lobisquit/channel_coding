import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def get_A(H):
    '''
    Obtain parity bits generating matrix, i.e. last
    (n-k) lines of G, below identity matrix of rank K

    Parameters
    ----------
    H : scipy.sparse.csc_matrix or np.array
        Parity check matrix of given code

    Returns
    -------
    scipy.sparse.csc_matrix
        Parity bits generating matrix A
    '''
    if isinstance(H, np.ndarray):
        # sparsify matrix if needed
        H = sp.csc_matrix(H)

    # H has shape (n-k) x n
    n = H.shape[1]
    k = n - H.shape[0]

    B = H[0:(n - k):1, 0:k:1]     # (n-k) x k
    C = H[0:(n - k):1, k:n:1]     # (n-k) x (n-k)

    A = spla.inv(C).dot(B)        # (n-k) x k

    return A

def get_generating_matrix(H):
    '''
    Convert parity check matrix H to its generating matrix G

    Parameters
    ----------
    H : scipy.sparse.csc_matrix or np.ndarray
        Parity check matrix of given code

    Returns
    -------
    scipy.sparse.csc_matrix
        Sparse encoding matrix for given code
    '''
    A = get_A(H)
    k = A.shape[1]
    G = sp.vstack((sp.eye(k), A)) # n x k

    return G

def encoder(H):
    '''
    Create encoder function from generating matrix

    Parameters
    ----------
    H : np.ndarray
        Generating matrix

    Returns
    -------
    callable
        Function that accepts vector u as input
        and returns its encoded vector c
    '''
    A = get_A(H)

    # here modulo is needed, since product could
    # lead to values different from 0 and 1
    return lambda u: np.concatenate( (u, A.dot(u) % 2) ).astype(int)
