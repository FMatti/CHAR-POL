import numpy as np
import scipy as sp
import sympy as sy


def baseline(A, d=500):
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

            p_A(x) = 5*x^3 - 9*x + 3   ===>   coeffs = [5, 0, 9, 3]
    """
    A, n = _verify_input(A, convert_to_numpy=True)

    # Compute characteristic polynomial symbolically
    charpol = sy.Matrix(A).charpoly(x='x')

    # Extract the coefficients of the characteristic polynomial to a numpy array
    coeffs = np.array([c.evalf(d) for c in charpol.coeffs()], dtype=np.float64)

    # In the leading coefficients are zero, we have to manually add them
    if (diff := n - len(coeffs)) >= 0:
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



def krylov(A, b=None, seed=42):
    """
    Computes the coefficients of the characteristic polynomial of a matrix A
    using the Krylov method. 

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix for which the characteristic polynomial is computed.
    b : int, optional, default is 42
        Vector used to generate the Krylov basis:

            K_n(A, b) = [b, A*b, A^2*b, ..., A^(n-1)*b]

    seed : int, optional, default is None
        If argument b is None, this seed is used to generate b randomly.

    Returns
    -------
    coeffs : numpy.ndarray
        The coefficients of the characteristic polynomial of A collected in
        descending order in a numpy array. Example:

            cp_A(x) = 5*x^3 - 9*x + 3   ===>   coeffs = [5, 0, 9, 3]
    """
    # Check if the matrix A is of a type compatible with the algorithm
    A, n = _verify_input(A, convert_to_numpy=False)

    # Generate a random vector, if no Krylov basis vector was given
    if b is None:
        b = np.random.randn(n)

    # Generate the Krylov basis
    K = np.empty((n, n))
    K[:, 0] = b
    for i in range(1, n):
        K[:, i] = A @ K[:, i-1]

    # Compute last column of the companion matrix (solve K c = A^n b)
    c = np.linalg.lstsq(K, A @ K[:, -1], rcond=None)[0]

    # Create coefficients vector based on last column of the companion matrix
    coeffs = np.append(1, -c[::-1])
    return coeffs


def hyman(A, b=None, seed=42):
    """
    Computes the coefficients of the characteristic polynomial of a matrix A
    using the Hyman's method.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix for which the characteristic polynomial is computed.
    b : int, optional, default is None
        Vector used to compute Arnoldi decomposition with.
    seed : int, optional, default is 42
        If argument b is None, this seed is used to generate b randomly.

    Returns
    -------
    coeffs : numpy.ndarray
        The coefficients of the characteristic polynomial of A.
    """
    # Check if the matrix A is of a type compatible with the algorithm
    A, n = _verify_input(A, convert_to_numpy=False)

    # Generate a random vector, if no Krylov basis vector was given
    if b is None:
        b = np.random.randn(n)

    # Compute Hessenberg matrix resulting from a full Arnoldi decomposition of A
    A_hessenberg = _arnoldi_decomposition(A, b)

    # Define the iteration matrix
    F = np.c_[-np.eye(n, 1, 0), A_hessenberg[:, :-1]]

    # Perform the iterative solution of upper triangular systems
    X = np.zeros((n+1, n+1))
    X[:-1, 0] = sp.linalg.solve_triangular(F, -A_hessenberg[:, -1], lower=False)
    X[-1, 0] += 1  # This is equivalent to adding g_n = [0, 0, ..., 0, 1]
    for i in range(1, n+1):
        # Shifting X's columns is same as multiplying with G = np.eye(n, n, 1)
        X[:-1, i] = sp.linalg.solve_triangular(F, X[1:, i-1], lower=False)

    # Extract coefficients
    coeffs = (-1)**(n+1) * X[0, ::-1] * np.prod(np.diag(A_hessenberg[1:]))
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
    A, n = _verify_input(A, convert_to_numpy=True)

    # Compue the eigenvalues of the matrix A
    eigenvalues = np.linalg.eig(A)[0]

    # Perform the summation algorithm to obtain the coefficients
    coeffs = np.append(1, np.zeros_like(eigenvalues))
    for i in range(n):
        coeffs[1:i+2] -= eigenvalues[i]*coeffs[:i+1]

    return coeffs


def _verify_input(A, convert_to_numpy=False):
    """Verify that the matrix A is of correct type"""
    if convert_to_numpy and isinstance(A, sp.sparse.spmatrix):
        A = A.toarray()
    elif not isinstance(A, np.ndarray | sp.sparse.spmatrix):
        msg = "Argument 'A' must be numpy.ndarray or scipy.sparse.spmatrix" \
              + ", but got {}.".format(type(A).__name__)
        raise TypeError(msg)
    return A, A.shape[0]


def _arnoldi_decomposition(A, b):
    """Compute Hessenberg matrix resulting from Arnoldi decomposition"""
    n = A.shape[0]

    U = np.empty((n, n+1))
    H = np.zeros((n+1, n))

    U[:, 0] = b / np.linalg.norm(b)
    for j in range(n):
        w = A @ U[:, j]
        H[:j+1, j] = U[:, :j+1].T @ w
        u_tilde = w - U[:, :j+1] @ H[:j+1, j]

        if np.linalg.norm(u_tilde) <= 0.7 * np.linalg.norm(w):
            # Twice is enough
            h_hat = U[:, :j+1].T @ u_tilde
            H[:j+1, j] += h_hat
            u_tilde -= U[:, :j+1] @ h_hat

        H[j+1, j] = np.linalg.norm(u_tilde)
        U[:, j+1] = u_tilde / H[j+1, j]

    return H[:-1]
