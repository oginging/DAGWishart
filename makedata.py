import numpy as np


def case2():
    p = 4
    n = 10
    trueL = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.7, 1.0, 0.0, 0.0],
        [0.0, 0.7, 1.0, 0.0],
        [-0.6, 0.0, -0.7, 1.0]
    ])
    pre = np.dot(trueL, trueL.T)
    sigma = np.linalg.inv(pre)
    Y = np.random.multivariate_normal(
        np.zeros(
            pre.shape[0]),
        np.linalg.inv(pre),
        n)  # 標本
    S = np.cov(Y, rowvar=0, bias=0)  # 標本不偏分散共分散行列
    # print(Y)
    # print(S)
    return trueL, pre, sigma, p, n, Y, S


def case3():
    p = 5
    n = 10
    sigma = np.eye(p)
    for i in range(p):
        for j in range(p):
            sigma[i, j] = 0.5 ** (abs(i - j))
    pre = np.linalg.inv(sigma)
    trueL = np.linalg.cholesky(pre)
    Y = np.random.multivariate_normal(
        np.zeros(
            pre.shape[0]),
        np.linalg.inv(pre),
        n)  # 標本
    S = np.cov(Y, rowvar=0, bias=0)  # 標本不偏分散共分散行列
    return trueL, pre, sigma, p, n, Y, S


def case4():
    p = 3
    n = 5
    trueL = np.array([
        [1.0, 0.0, 0.0],
        [0.7, 1.0, 0.0],
        [-0.6, 0.4, 1.0],
    ])
    pre = np.dot(trueL, trueL.T)
    sigma = np.linalg.inv(pre)
    Y = np.random.multivariate_normal(
        np.zeros(
            pre.shape[0]),
        np.linalg.inv(pre),
        n)  # 標本
    S = np.cov(Y, rowvar=0, bias=0)  # 標本不偏分散共分散行列
    # print(Y)
    # print(S)
    return trueL, pre, sigma, p, n, Y, S


def case5():
    p = 35
    n = 100
    trueL = np.eye(p)
    for i in range(p):
        for j in range(i):
            rn = np.random.rand()
            if rn <= 0.10:
                trueL[i, j] = 0.5 + 0.5 * np.random.rand()
    # print(trueL)
    pre = np.dot(trueL, trueL.T)
    sigma = np.linalg.inv(pre)
    Y = np.random.multivariate_normal(
        np.zeros(
            pre.shape[0]),
        np.linalg.inv(pre),
        n)  # 標本
    S = np.cov(Y, rowvar=0, bias=0)  # 標本不偏分散共分散行列
    # print(Y)
    # print(S)
    return trueL, pre, sigma, p, n, Y, S
