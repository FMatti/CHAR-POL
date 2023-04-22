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
    """
    Computes the coefficients of the characteristic polynomial 
    of an n x n matrix A using Leverrier’s method:

        p(x) := det(x*I - A) = c_0 * x^n + c_1 * x^(n - 1) + ... + c_n

    The property of p(x) is that: 

        c_0 = 1, c_n = (-1)^n * det(A)

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix for which the characteristic polynomial is computed.

    Returns
    -------
    coeffs : numpy.ndarray
        The coefficients of the characteristic polynomial of A: coeffs = [c_0, c_1, ..., c_n]
    """
    n = A.shape[0]
    coeffs = []
    coeffs.append(1.)
    Bk = A.astype(np.float64)

    for k in range(1, n + 1):
        # calculate the k_th coefficient (Newton's identity)
        ck = -Bk.trace() / k
        # add the coefficient to the solution
        coeffs.append(ck.item())

        Bk += np.multiply(ck, np.identity(n))
        Bk = A @ Bk
    return np.array(coeffs)



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
    """
    Computes the coefficients of the characteristic polynomial of a matrix A
    using the Hyman’s method for Hessenberg matrices. 

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix for which the characteristic polynomial is computed.

    Returns
    -------
    coeffs : numpy.ndarray
        The coefficients of the characteristic polynomial of A.
    """
    n = len(A)

    # compose matrix F and a vector f_n
    F = np.zeros((n,n))
    ut_A = np.zeros((n,n))

    uti = np.triu_indices(n, -1)
    uti_n = np.triu_indices(n)
    ut_A[uti] = A[uti]
    F[uti_n[0][1:], uti_n[1][1:]] = ut_A[np.triu_indices(n,-1,n-1)]
    F[0, 0] = -1

    f_n = A[:,-1]

    # compose matrix G and a vector g_n
    G = np.diag(np.ones(n-1),1)
    g_n = np.zeros(n)
    g_n[n-1] = 1

    X = np.zeros((n, n+1))
    X[:, 0] = np.linalg.solve(F, - f_n)
    X[:, 1] = np.linalg.solve(F, G.dot(X[:, 0]) + g_n)
    
    for i in range(2,n+1):
        X[:, i] = np.linalg.solve(F, G.dot(X[:, i-1]) + g_n)

    # TODO: understand how to extract coeffs 
    coeffs = None

    return coeffs


def summation(A):
    """
    Computes the coefficients of the characteristic polynomial of a matrix A
    using the summation method based on the eigenvalues. 

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix for which the characteristic polynomial is computed.

    Returns
    -------
    coeffs : numpy.ndarray
        The coefficients of the characteristic polynomial of A collected in
        descending order in a numpy array. Example:

            cp_A(x) = 5*x^3 - 9*x + 3   ===>   coeffs = [5, 0, 9, 3]
    """
    A = _verify_input(A, convert_to_numpy=True)
    n = len(A)

    # Compue the eigenvalues of the matrix A
    eigenvalues = np.linalg.eig(A)[0]

    # Apply the summation algorithm to obtain the elementary functions
    s = np.zeros(n+1, dtype=eigenvalues.dtype)
    s[0] = 1
    s[1] = eigenvalues[0]

    for i in range(2, n+1):
        s[1:i+1] += eigenvalues[i-1]*s[:i]

    # Deduce the coefficients of the characteristic polynomial
    coeffs = s * (-1)**np.arange(n+1)
    return coeffs


def _verify_input(A, convert_to_numpy=False):
    if convert_to_numpy and isinstance(A, sp.sparse.spmatrix):
        A = A.todense()
    elif not isinstance(A, np.ndarray | sp.sparse.spmatrix):
        msg = "Argument 'A' must be numpy.ndarray or scipy.sparse.spmatrix" \
              + ", but got {}.".format(type(A).__name__)
        raise TypeError(msg)
    return A