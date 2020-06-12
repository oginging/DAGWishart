import numpy as np
from math import sqrt
from math import log


def s_lam(x, lam):
    if x >= lam:
        return x - lam
    elif x <= -lam:
        return x + lam
    else:
        return 0.0


def update_x_at_j(A, x, j, lam):
    if A.shape[0] != x.size:
        print(12345678)
    k = x.size
    if j > 0:
        val = 0.0
        for l in range(k):
            if l != j:
                val -= 2 * A[l, j] * x[l]
        return s_lam(val, lam) / (2 * A[j, j])
    else:
        val1 = 0.0
        for l in range(k):
            if l != j:
                val1 -= A[l, j] * x[l]
        return (val1 + sqrt(val1 * val1 + 4 * A[j, j])) / (2 * A[j, j])


def update_x(A, x, lam):
    x_old = x.copy()
    k = x.size
    for j in range(k - 1, -1, -1):
        x[j] = update_x_at_j(A, x, j, lam)
    e = x - x_old
    max_change = np.max(np.abs(e))
    # print("max_change: ", max_change )
    return x, max_change


def calc_x(A, x, lam, maxiter=1000, eps=0.0001):
    while maxiter > 0:
        maxiter -= 1
        x, e = update_x(A, x, lam)
        if e < eps:
            # print("iter", 1000-maxiter)
            break
    return x


def calc_L(S, L0, lam):
    L = L0.copy()
    for i in range(S.shape[0]):
        x = L[i:, i]
        x = calc_x(S[i:, i:], x, lam)
        L[i:, i] = x
    return L


def makeDAG(L):
    p = L.shape[0]
    edges = 0
    DAG = 2 * np.eye(p)
    for j in range(p):
        for i in range(j):
            if L[j, i] != 0.0:
                DAG[j, i] = 1
                edges += 1
    return DAG, edges


def numedges(L):
    p = L.shape[0]
    edges = 0
    for j in range(p):
        for i in range(p):
            if L[j, i] != 0.0:
                edges += 1
    return edges


def BIC(n, L, S):  # Stein's loss
    DAG, edges = makeDAG(L)
    A = np.dot(np.dot(L, L.T), S)
    # print("BIC", n * np.trace(A) - 2 * n * log(np.linalg.det(L)) + log(n) * edges)
    return n * np.trace(A) - 2 * n * log(np.linalg.det(L)) + log(n) * edges


def ichol(L, D):
    return np.dot(L, np.dot(np.linalg.inv(D), L.T))


def BIC2(n, L, D, S):  # Stein's loss
    Omega = ichol(L, D)
    edges = numedges(L) + numedges(Omega)
    A = np.dot(Omega, S)
    if np.linalg.det(Omega) == 0:
        return 202020202020
    # print("BIC", n * np.trace(A) - 2 * n * log(np.linalg.det(L)) + log(n) * edges)
    return n * np.trace(A) - n * log(np.linalg.det(Omega)) + log(n) * edges


def threshold(L, la):
    for i in range(L.shape[0]):
        for j in range(L.shape[0]):
            if abs(L[i, j]) <= la:
                L[i, j] = 0.0
    return L
