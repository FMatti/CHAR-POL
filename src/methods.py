import numpy as np
import scipy as sp
import sympy as sy


def baseline(A, d=16):
    """
    Computes the coefficients of the characteristic polynomial of a matrix A
    symbolically using the SymPy python library. 

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix for which the characteristic polynomial is computed.
    d : int
        Number of decimal digits computed in numerical evaluation.

    Returns
    -------
    coeffs : numpy.ndarray
        The coefficients of the characteristic polynomial of A collected in
        descending order in a numpy array. Example:

            cp_A(x) = 5*x^3 - 9*x + 3   ===>   coeffs = [5, 0, 9, 3]
    """
    A = _verify_input(A, convert_to_numpy=True)
    charpol = sy.Matrix(A).charpoly(x='x')
    coeffs = np.array([c.evalf(d) for c in charpol.coeffs()], dtype=np.float64)
    return coeffs


def leverrier(A):
    """
    Given an n x n matrix A, this function returns the coefficients of it's
    characteristic polynomial:

        p(x) := det(x*I - A) = c_0 * x^n + c_1 * x^(n - 1) + ... + c_n

    The property of p(x) is that: 

        c_0 = 1, c_n = (-1)^n * det(A)

    Parameters
    ----------
    A : np.array, shape (n, n)
        Matrix A.

    Returns
    -------
    c : np.array, 
        The coefficients of the characteristic polynomial: c = [c_0, c_1, ..., c_n]
    """
    n = A.shape[0]
    c = []
    c.append(1.)
    Bk = A.astype(np.float64)

    for k in range(1, n + 1):
        # calculate the k_th coefficient (Newton's identity)
        ck = -Bk.trace() / k
        # add the coefficient to the solution
        c.append(ck.item())

        Bk += np.multiply(ck, np.identity(n))
        Bk = A @ Bk
    return np.array(c)



def krylov(A):
    raise NotImplementedError("This method still needs to be implemented.")
    return coeffs


def hyman(A):
    
    return coeffs


def summation(A):
    raise NotImplementedError("This method still needs to be implemented.")
    return coeffs


def _verify_input(A, convert_to_numpy=False):
    if convert_to_numpy and isinstance(A, sp.sparse.spmatrix):
        A = A.todense()
    elif not isinstance(A, np.ndarray | sp.sparse.spmatrix):
        msg = "Argument 'A' must be numpy.ndarray or scipy.sparse.spmatrix" \
              + ", but got {}.".format(type(A).__name__)
        raise TypeError(msg)
    return A