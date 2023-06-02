import numpy as np
import scipy as sp


def A_1(n):
    """
    Obtain the rank-1 modification of the tridiagonal Topelitz matrix.

    Parameters
    ----------
    n : int, n >= 1
        Size of the matrix A.

    Returns
    -------
    A : scipy.sparse._dia.dia_matrix, shape (n, n)
        The generated matrix in a scipy sparse diagonal format.
    """
    _verify_input(n)
    diags = np.ones((3, n), dtype=int)
    diags[[0, 2]] *= -1
    diags[1, 1:] *= 2
    A = sp.sparse.spdiags(diags, (-1, 0, 1), m=(n, n))
    return A


def A_2(n, seed=42):
    """
    Obtain a random tridiagonal matrix.

    Parameters
    ----------
    n : int, n >= 1
        Size of the matrix A.
    seed : int, optional, default is 42
        Seed for the random number generator.

    Returns
    -------
    A : scipy.sparse._dia.dia_matrix, shape (n, n)
        The generated matrix in a scipy sparse diagonal format.
    """
    _verify_input(n)
    np.random.seed(seed)
    diags = np.random.randn(3, n)
    A = sp.sparse.spdiags(diags, (-1, 0, 1), m=(n, n))
    return A


def A_3(n):
    """
    Obtain the Frank matrix.

    Parameters
    ----------
    n : int, n >= 1
        Size of the matrix A.

    Returns
    -------
    A : numpy.ndarray, shape (n, n)
        The generated matrix as a numpy array.
    """
    _verify_input(n)
    A = np.repeat(np.arange(n, 0, -1, dtype=int)[None], repeats=n, axis=0)
    A = np.triu(A, k=-1) - np.diag(np.ones(n - 1, dtype=int), k=-1)
    return A


def _verify_input(n):
    if not isinstance(n, int):
        msg = "Argument 'n' must be int, but got {}.".format(type(n).__name__)
        raise TypeError(msg)
    if n < 1:
        msg = "Argument 'n' must be >= 1, but got {}.".format(n)
        raise ValueError(msg)
