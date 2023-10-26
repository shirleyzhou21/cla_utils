import numpy as np
import timeit
import numpy.random as random

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(500, 500)
x0 = random.randn(500)


def basic_matvec(A, x):
    m, n = A.shape
    b = np.zeros(m)
    for i in range(m):
        for j in range(n):
            b[i] += A[i,j] * x[j]
    return b 


def column_matvec(A, x):
    m, n = A.shape
    b = np.zeros(m)
    for j in range(n):
        b += A[:, j] * x[j]
    return b

    """
    Matrix-vector multiplication using the representation of the product
    Ax as linear combinations of the columns of A, using the entries in 
    x as coefficients.


    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    :return b: an m-dimensional numpy array which is the product of A with x

    This should be implemented using a single loop over the entries of x
    """


def timeable_basic_matvec():
    """
    Doing a matvec example with the basic_matvec that we can
    pass to timeit.
    """

    b = basic_matvec(A0, x0) # noqa


def timeable_column_matvec():
    """
    Doing a matvec example with the column_matvec that we can
    pass to timeit.
    """

    b = column_matvec(A0, x0) # noqa


def timeable_numpy_matvec():
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """

    b = A0.dot(x0) # noqa


def time_matvecs():
    """
    Get some timings for matvecs.
    """

    print("Timing for basic_matvec")
    print(timeit.Timer(timeable_basic_matvec).timeit(number=1))
    print("Timing for column_matvec")
    print(timeit.Timer(timeable_column_matvec).timeit(number=1))
    print("Timing for numpy matvec")
    print(timeit.Timer(timeable_numpy_matvec).timeit(number=1))
time_matvecs()
"""
Timing for basic_matvec
0.06561495800269768
Timing for column_matvec
0.001418125000782311
Timing for numpy matvec
0.0037960830377414823
"""

def rank2(u1, u2, v1, v2):
    A = np.outer(u1, np.conj(v1)) + np.outer(u2, np.conj(v2))
    return A


def rank1pert_inv(u, v):
    m = len(u)
    uv = np.outer(u, np.conj(v))
    I = np.eye(m)
    A = I + uv
    Ainv = np.linalg.inv(A)
    return Ainv

def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, where A = B + iC
    with

    :param Ahat: an mxm-dimensional numpy array with Ahat[i,j] = B[i,j] \
    for i>=j and Ahat[i,j] = C[i,j] for i<j.

    :return zr: m-dimensional numpy arrays containing the real part of z.
    :return zi: m-dimensional numpy arrays containing the imaginary part of z.
    """
    m, _ = Ahat.shape
    zr = np.zeros(m)
    zi = np.zeros(m)
    for j in range(m):
        zr[j] = sum(Ahat[j, k] * xr[k] for k in range(j)) - sum(Ahat[j, k] * xi[k] for k in range(j + 1, m))
        zi[j] = sum(Ahat[j, k] * xi[k] for k in range(j)) + sum(Ahat[j, k] * xr[k] for k in range(j + 1, m))
    return zr, zi




