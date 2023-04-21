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
    if (diff := len(A) - len(coeffs)) >= 0:
        coeffs = np.append(coeffs, np.zeros(diff + 1))
    return coeffs


def leverrier(A):
    raise NotImplementedError("This method still needs to be implemented.")


def krylov(A, b=None):
    """
    Computes the coefficients of the characteristic polynomial of a matrix A
    using the Krylov method. 

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix for which the characteristic polynomial is computed.
    b : int
        Vector used to generate the Krylov basis:

            [b, A*b, A^2*b, ..., A^(k-1)*b]

    Returns
    -------
    coeffs : numpy.ndarray
        The coefficients of the characteristic polynomial of A collected in
        descending order in a numpy array. Example:

            cp_A(x) = 5*x^3 - 9*x + 3   ===>   coeffs = [5, 0, 9, 3]
    """
    A = _verify_input(A, convert_to_numpy=True)
    n = len(A)

    # Generate a random vector, if no Krylov basis vector was given
    if b is None:
        b = np.random.randn(n)

    # Generate the Krylov basis
    K = np.empty((n, n))
    K[:, 0] = b
    for i in range(1, n):
        K[:, i] = A @ K[:, i-1]

    # Compute the companion matrix
    companion_matrix = np.linalg.solve(K, A @ K)

    # Derive coefficients from last column of companion matrix
    coeffs = np.append(1, -companion_matrix[::-1, -1])
    return coeffs


def hyman(A):
    raise NotImplementedError("This method still needs to be implemented.")
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