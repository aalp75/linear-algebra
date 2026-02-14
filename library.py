import math
import numpy as np

def is_sym(a):
    """
    input : numpy a
    output : boolean if the matrix a is symmetric
    """
    n, m = a.shape[0], a.shape[1]
    if n != m:
        print("Not squared Matrix")
        return False
    for i in range(n // 2):
        for j in range(n // 2):
            if a[i, j] != a[j, i]:
                return False
    return True


def determinant_naive(a):
    """
    input : numpy a
    output : float
    
    calculate the determinant of a matrix a
    iterative method
    """
    det = 0
    if a.shape[0] != a.shape[1]:
        print("Not squared matrix")
        return 0
    n = a.shape[0]
    if n == 2:
        det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
        return det
    index = np.arange(n)
    for i in range(n):
        if a[0, i] != 0:
            det += pow(-1, i) * a[0, i] * determinant_naive(a[1:n, index != i])
    return det


def permute_rows(a, i1, i2):
    n = a.shape[0]
    tmp = np.zeros(n)
    for j in range(n):
        tmp[j] = a[i1, j]
    a[i1, :] = a[i2, :]
    for j in range(n):
        a[i2, j] = tmp[j]
    return a


def determinant_gauss(a, show=False):
    """
    in : numpy a
    out : float
    calculate the determinant of a matrix a
    gaussian elimination method
    show=True if you want to print the tringular matrix at the end
    """
    a = a.astype(float)
    det = 0
    n, m = a.shape[0], a.shape[1]
    if n != m:
        print("Not squared Matrix")
        return 0
    number_permut = 0
    for j in range(n):
        row = n
        for i in range(j, n):
            if a[i, j] != 0:
                row = i
                break
        if row == n:
            return 0
        if row != j:
            permute_rows(a, j, row)
            number_permut += 1
        for i in range(j + 1, n):
            mult = a[i, j] / a[j, j]
            a[i, :] = a[i, :] - a[j, :] * mult
    if show:
        print(a)
    det = 1
    for i in range(n):
        det = det * a[i, i]
    return pow(-1, number_permut) * det


def is_def_pos_sylvester(a):
    """
    input : numpy a, a symmetric matrix
    output : boolean if the matrix is positive definite
    Sylvester criterion that checks if all the minor determinants are positive
    """
    n, m = a.shape[0], a.shape[1]
    if n != m:
        print("Not squared Matrix")
        return False
    for i in range(1, n + 1):
        if determinant_gauss(a[0:i, 0:i]) <= 0:
            return False
    return True


def inverse_gauss(a):
    """
    input : numpy a, a squared matrix
    output : the inverse of a
    inverse by Gaussian elimination
    """
    a = a.astype(float)
    n, m = a.shape[0], a.shape[1]
    if n != m:
        print("Not squared Matrix")
        return np.zeros((n, n))
    inv = np.identity(n)
    for j in range(n):
        row = n
        for i in range(j, n):
            if a[i, j] != 0:
                row = i
                break
        if row == n:
            return 0
        if row != j:
            permute_rows(a, j, row)
            permute_rows(inv, j, row)
        for i in range(j + 1, n):
            mult = a[i, j] / a[j, j]
            a[i, :] = a[i, :] - a[j, :] * mult
            inv[i, :] = inv[i, :] - inv[j, :] * mult
    det = 1
    for i in range(n):
        det = det * a[i, i]
    if det == 0:
        print("Not inversible")
        return np.zeros((n))
    for j in np.arange(n - 1, 0, -1):
        divisor = a[j, j]
        a[j, j] = a[j, j] / divisor
        inv[j, :] = inv[j, :] / divisor
        for i in np.arange(j - 1, -1, -1):
            mult = a[i, j]
            a[i, :] = a[i, :] - mult * a[j, :]
            inv[i, :] = inv[i, :] - mult * inv[j, :]
    divisor = a[0, 0]
    a[0, 0] = a[0, 0] / divisor
    inv[0, :] = inv[0, :] / divisor
    return inv


def cholesky(a):
    """
    Input : numpy a(n, n) (symmetric matrix)
    Output : numpy L(n, n), the triangular lower matrix of the Cholesky decomposition
    a = L * L.T
    """
    if not is_def_pos_sylvester(a):
        print("Not squared symetric matrix")
        return 0
    n = a.shape[0]
    l = np.zeros((n, n))
    l[0, 0] = math.sqrt(a[0, 0])
    for j in range(1, n):
        l[j, 0] = a[0, j] / l[0, 0]
    for i in range(1, n):
        l[i, i] = math.sqrt(a[i, i] - pow(l[i, 0:i], 2).sum())
        for j in range(i + 1, n):
            l[j, i] = (a[i, j] - np.dot(l[i, 0:i].T, l[j, 0:i])) / l[i, i]
    return l


def determinant_cholesky(a):
    l = cholesky(a)
    determinant = 1
    for i in range(l.shape[0]):
        determinant = determinant * pow(l[i, i], 2)
    return determinant


def power_iteration(a, num_simulations_max=pow(10, 5), tol=pow(10, -3)):
    """
    input : a, a squared matrix
    output : the largest eigenvalue and the associated eigenvector
    power iteration method x = a*a*a*a*a*...*a(x)
    """
    x_n = np.random.rand(a.shape[1])

    for i in range(num_simulations_max):
        x_n1 = np.dot(a, x_n)
        x_n1_norm = np.linalg.norm(x_n1, 2)
        x_n = x_n1 / x_n1_norm
        if i > 200 and np.linalg.norm(abs(x_n) - abs(x_old), 2) < tol:
            break

        x_old = x_n
        if i == num_simulations_max - 1:
            return [None, None]
    eigenvalue = round(np.dot(np.dot(x_n.T, a), x_n) / np.dot(x_n.T, x_n), 5)
    res = [eigenvalue, x_n]

    return res


def deflation(a, eigenvalue, v, w):
    """
    v is the eigenvector of a associated with the eigenvalue lambda
    w is the eigenvector of a.T associated with the eigenvalue lambda
    B = a - lambda * (v*w.T)/(w.T,v)
    """
    v = v.reshape(a.shape[0], 1)
    w = w.reshape(a.shape[0], 1)
    b = a - eigenvalue * np.dot(v, w.T) / np.dot(w.T, v)
    return b


def eigen(a):
    """
    input : numpy squared matrix a
    out : eigenvalues and eigenvectors of the matrix if a is diagonalizable
    """
    b = a
    if b.shape[0] != b.shape[1]:
        print("Not squared Matrix")
        return (0, 0)
    eigenvectors = []
    eigenvalues = []
    for i in range(b.shape[0]):
        tmp = power_iteration(b)
        if tmp[0] is None:
            print("Matrix is not diagonalizable")
            break
        eigenvalues.append(tmp[0])
        eigenvectors.append(tmp[1])
        tmp2 = power_iteration(b.T)
        b = deflation(b, tmp[0], tmp[1], tmp2[1])
    return eigenvalues, eigenvectors


def eigen_matrix(a):
    """
    input : numpy squared matrix a
    out : the transition matrix P if a is diagonalizable such that a = inv(P)*D*P
    """
    eigenvalues, eigenvectors = eigen(a)
    n = a.shape[0]
    p = np.zeros((n, n))
    for i in range(n):
        p[:, i] = eigenvectors[i]
    return p
