import numpy as np

def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """
    u = np.dot(Q.conj().T, v)
    r = v - np.dot(Q, u)
    return r, u



def solve_Q(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """
    Q_star = np.conj(Q.T)
    x = np.dot(Q_star, b)
    return x


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """
    P = np.dot(Q, Q.conj().T)
    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an mxl-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U, for appropriate l.
    """
    Q, R = np.linalg.qr(V, mode='complete')
    m, n = V.shape
    Q = Q[:, n:]
    return Q


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place and returning R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """
    m, n = A.shape
    R = np.zeros((n, n), dtype=complex)
    for j in range(n):
        vj = A[:, j]
        for i in range(j):
            rij = np.dot(A[:, i].conj(), A[:, j])
            R[i, j] = rij
            vj -= rij * A[:, i]
        R[j, j] = np.linalg.norm(vj)
        A[:, j] = vj / R[j, j]
    return R

def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and returning
    R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """
    m, n = A.shape
    R = np.zeros((n, n), dtype=complex)
    for i in range(n):
        R[i, i] = np.linalg.norm(A[:, i])
        A[:, i] /= R[i, i]
        for j in range(i + 1, n):
            R[i, j] = np.dot(A[:, i].conj(), A[:, j])
            A[:, j] -= R[i, j] * A[:, i]
    return R


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is normalised and orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """
    m, n = A.shape
    R = np.eye(n, dtype=complex)
    for j in range(k, n):
        for i in range(k):
            R[i, j] = np.dot(A[:, i].conj(), A[:, j])
            A[:, j] -= R[i, j] * A[:, i]
        R[k, j] = np.linalg.norm(A[:, j])
        A[:, j] /= R[k, j]
    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """
    m, n = A.shape
    R = np.zeros((n, n), dtype=complex)
    for k in range(n):
        R[k, k] = np.linalg.norm(A[:, k])
        A[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, n):
            R[k, j] = np.dot(A[:, k].conj(), A[:, j])
            A[:, j] -= R[k, j] * A[:, k]
    return A, R
