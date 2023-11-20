import numpy as np


def householder(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations. The reduction should be done "in-place",
    so that A is transformed to R.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.
    """

    m, n = A.shape
    if kmax is None:
        kmax = n
    for k in range(kmax):
        x = A[k:m, k]
        vk = np.sign(x[0]) * np.linalg.norm(x) * np.eye(m-k)[0] + x
        vk = vk / np.linalg.norm(vk)
        A[k:m, k:n] = A[k:m, k:n] - 2 * np.outer(vk, np.dot(vk.conj(), A[k:m, k:n]))


def solve_U(U, b):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """
    m, k = b.shape
    x = np.zeros((m, k))
    for i in range(m-1, -1, -1):
        for j in range(k):
            x[i, j] = (b[i, j] - np.dot(U[i, i+1:], x[i+1:, j])) / U[i, i]
    return x
    



def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """
    m, n = A.shape
    k = b.shape[1]
    extended_A = np.hstack((A, b))  
    A_copy = extended_A.copy() 
    householder(A_copy, kmax=n)  
    new_b = A_copy[:, n:] 
    R = A_copy[:, :n]  
    x = solve_U(R, new_b)
    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the full QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """
    m, n = A.shape
    Q = np.eye(m)
    R = np.array(A, copy=True)  

    for k in range(n):
        x = R[k:m, k]
        vk = np.sign(x[0]) * np.linalg.norm(x) * np.eye(m-k)[0] + x
        vk = vk / np.linalg.norm(vk)
        Hk = np.eye(m-k) - 2 * np.outer(vk, vk)
        H = np.eye(m)
        H[k:m, k:m] = Hk
        R = np.dot(H, R)
        Q = np.dot(Q, H.T)

    return Q, R


def householder_ls(A, b):
    """
    
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """
    A = np.array(A)  
    b = np.array(b).reshape(-1, 1) 
    
    augmented_matrix = np.hstack((A, b))

    householder(augmented_matrix)
    
    n = A.shape[1]
    R = augmented_matrix[:n, :n]
    b_augmented = augmented_matrix[:n, n]
    
    x = solve_U(R, b_augmented.reshape(-1, 1))

    return x
