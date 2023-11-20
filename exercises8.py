import numpy as np
import cla_utils

def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """
    m, n = A.shape
    assert m == n

    x = A[:, 0]
    e1 = np.zeros_like(x)
    e1[0] = 1
    v = x + np.sign(x[0]) * np.linalg.norm(x) * e1
    v = v / np.linalg.norm(v)
    
    Q1 = np.eye(m) - 2 * np.outer(v, v)
    
    Q1A = Q1 @ A
    A1 = Q1A @ Q1.T
    return A1


def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """
    m, n = A.shape
    assert m == n
    
    for k in range(m - 2):
        x = A[k+1:m, k]
        e1 = np.zeros_like(x)
        e1[0] = 1
        vk = x + np.sign(x[0]) * np.linalg.norm(x) * e1
        vk = vk / np.linalg.norm(vk)

        A[k+1:m, k:n] -= 2 * np.outer(vk, vk.T @ A[k+1:m, k:n])
        A[0:m, k+1:m] -= 2 * np.outer(A[0:m, k+1:m] @ vk, vk.T)
    return A



def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """
    m, n = A.shape
    assert m == n
    Q = np.eye(m)
    
    for k in range(n-2):
        x = A[k+1:, k]
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = x + np.sign(x[0]) * np.linalg.norm(x) * e1
        v = v / np.linalg.norm(v)

        Qk = np.eye(m)  # Start with an identity matrix of size m x m
        Qk_sub = np.eye(m-k-1) - 2 * np.outer(v, v)  # The sub-matrix for the Householder reflection
        Qk[k+1:, k+1:] = Qk_sub  # Embed the sub-matrix into the larger identity matrix
        A[k+1:, k:] = Qk_sub @ A[k+1:, k:]
        A[:, k+1:] = A[:, k+1:] @ Qk_sub.T
        Q = Q @ Qk
    return Q

def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvectors.

    :param H: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of H

    Do not change this function.
    """
    m, n = H.shape
    assert(m==n)
    assert(cla_utils.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """
    H = hessenbergQ(A)  
    V_H = hessenberg_ev(A)
    V = H @ V_H
    
    return V

