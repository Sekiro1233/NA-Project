import numpy as np
from scipy.sparse import random
from scipy.sparse.linalg import eigsh

# 题目1：生成随机矩阵******************************************************************************
print("题目1: ")
# (1)生成一个10x10的随机矩阵
matrix1 = np.random.rand(10, 10) * 10

# (2)生成一个10000x10000维度且密度为0.001的随机稀疏矩阵，并计算非零元素的数量
matrix2 = random(10000, 10000, density=0.001) * 10
nonzero_cnt = matrix2.count_nonzero()
print("(2)")
print("非零元素数量: ", nonzero_cnt)

# (3)计算matrix1和matrix2的特征值
eigenvalues1, _ = np.linalg.eig(matrix1)
eigenvalues2, _ = eigsh(matrix2, k=10)  # 当矩阵过大时，使用eigsh计算特征值更快，这里求最大的10个特征值
print("(3)")
print("矩阵(1)特征值: ", eigenvalues1)
print("矩阵(2)特征值: ", eigenvalues2)

# 题目2：给出 Power Method 的伪代码并⽤代码实现，能够输出绝对值最⼤的特征值********************************************************************************
print("题目2: ")


def power_method(matrix, tol, max_iter):
    k = 1
    n = matrix.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x, np.inf)  # 归一化，使得||x||∞ = 1
    while k <= max_iter:
        p = np.argmin(np.abs(x))
        x = x / x[p]
        y = matrix @ x
        p = np.argmin(np.abs(y))
        if y[p] == 0:
            return "Eigenvector", x
        mu = y[p]
        err = np.linalg.norm(x - y / y[p], np.inf)
        x = y / y[p]
        if err < tol:
            return mu, x
        k += 1
    return "The maximum number of iterations exceeded"


# (1)利用power method求解matrix1中绝对值最大的特征值
eigenvalue_pm1, _ = power_method(matrix1, 1e-6, 1000)
print("(1)")
print("矩阵(1)绝对值最大的特征值: ", eigenvalue_pm1)

# (2)利用power method求解matrix2中绝对值最大的特征值
eigenvalue_pm2, _ = power_method(matrix2, 1e-6, 1000)
print("(2)")
print(
    "矩阵(2)绝对值最大的特征值: ", eigenvalue_pm2
)  # matrix2是一个很大的稀疏矩阵，而power method算法在处理稀疏矩阵时可能会出现问题，因此这里的结果可能不准确

# 题目3：给出 QR 算法的伪代码并⽤代码实现，并能够实现输出前k个绝对值最⼤的特征值，其中k为⾃定义参数**************************************
print("题目3: ")


# 首先实现Householder变换
import numpy as np


def householder(a):
    n = a.shape[0]
    v = np.zeros((n, 1))
    p = np.zeros((n, n))
    a1 = np.zeros((n, n))

    for k in range(1, n - 1):
        r = 0
        for l in range(k, n):
            r = r + a[k - 1, l] * a[k - 1, l]
        r = np.sqrt(r)
        if r * a[k - 1, k] > 0:
            r = -r
        h = -1.0 / (r * r - r * a[k - 1, k])
        v[:] = 0
        v[k, 0] = a[k - 1, k] - r
        for l in range(k + 2, n + 1):
            v[l - 1, 0] = a[k - 1, l - 1]
        p = np.dot(v, np.transpose(v)) * h
        for l in range(1, n + 1):
            p[l - 1, l - 1] = p[l - 1, l - 1] + 1.0
        a1 = np.dot(p, a)
        a = np.dot(a1, p)

    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1:
                a[i, j] = 0

    result = []
    for i in range(1, n + 1):
        result.append(a[i - 1, i - 1])

    for i in range(2, n + 1):
        result.append(a[i - 2, i - 1])

    return a


matrix3 = np.array([[4, 1, -2, 2], [1, 2, 0, 1], [-2, 0, 3, -2], [2, 1, -2, -1]])
print(householder(matrix3))


# 基于Householder变换实现QR法求解特征值
def qr(matrix, tol, max_iter):
    k = 0
    sft = 0
    dd = [0] * n
    zz = [0] * n
    cc = [0] * n
    sigg = [0] * n
    qq = [0] * n
    xx = [0] * n
    yy = [0] * n
    while k <= max_iter:
        if matrix[n - 2, n - 1] <= tol:
            lam = matrix[n - 1, n - 1] + sft
            print("lambda: ", lam)
            n = n - 1
        if matrix[1, 0] <= tol:
            lam = matrix[0, 0] + sft
            print("lambda: ", lam)
            n = n - 1
            matrix[0, 0] = matrix[1, 1]
            for j in range(1, n):
                matrix[j, j] = matrix[j + 1, j + 1]
                matrix[j, j - 1] = matrix[j + 1, j]
        if n == 0:
            return ""
        if n == 1:
            lam = matrix[0, 0] + sft
            print("lambda: ", lam)
            return ""
        for j in range(2, n - 1):
            if matrix[j, j - 1] <= tol:
                print("split into ... and ...", sft)
        b = -(matrix[n - 2, n - 2] + matrix[n - 1, n - 1])
        c = (
            matrix[n - 2, n - 2] * matrix[n - 1, n - 1]
            - matrix[n - 2, n - 1] * matrix[n - 1, n - 2]
        )
        d = np.sqrt(b**2 - 4 * c)
        if b > 0:
            mu1 = -2 * c / (b + d)
            mu2 = -(b + d) / 2
        else:
            mu1 = (d - b) / 2
            mu2 = -2 * c / (d - b)
        if n == 2:
            lam1 = mu1 + sft
            lam2 = mu2 + sft
            print("lambda: ", lam1, lam2)
            return ""
        sig = (
            min(abs(mu1 - matrix[n - 1, n - 1]), abs(mu2 - matrix[n - 1, n - 1]))
            + matrix[n - 1, n - 1]
        )
        sft = sft + sig
        for j in range(0, n):
            dd[j] = matrix[j, j] - sig
        xx[0] = dd[0]
        yy[0] = matrix[1, 0]
        for j in (1, n):
            zz[j - 1] = np.sqrt(xx[j - 1] ** 2 + matrix[j - 1, j] ** 2)
            cc[j] = xx[j - 1] / zz[j - 1]
            sigg[j] = matrix[j - 1, j] / zz[j - 1]
            qq[j - 1] = cc[j] * yy[j - 1] + sigg[j] * dd[j]
            xx[j] = -sigg[j] * yy[j - 1] + cc[j] * dd[j]
            if j != n:
                yy[j] = matrix[j + 1, j] * cc[j]
        zz[n - 1] = xx[n - 1]
        matrix[0, 0] = sigg[1] * qq[0] + cc[1] * zz[0]
        matrix[1, 0] = sigg[1] * zz[1]
        for j in range(1, n - 1):
            matrix[j, j + 1] = sigg[j + 1] * zz[j + 1]
            matrix[j, j] = sigg[j + 1] * qq[j] + cc[j] * cc[j + 1] * zz[j]
            matrix[j + 1, j] = sigg[j + 1] * zz[j + 1]
        matrix[n - 1, n - 1] = cc[n - 1] * zz[n - 1]
        k += 1
    return "The maximum number of iterations exceeded"


z
# def qr(matrix, tol=1e-6, max_iter=1000):
#     n = matrix.shape[0]
#     sft = 0
#     k = 0
#     while k < max_iter:
#         q, r = np.linalg.qr(matrix)  # Compute QR decomposition
#         matrix = np.dot(r, q)  # Form A_{k+1}
#         if np.allclose(np.diag(matrix, -1), 0, atol=tol):
#             return np.diag(matrix) + sft
#         sft += np.trace(matrix) / n
#         k += 1
#     raise ValueError("The maximum number of iterations exceeded")

matrix_h = householder(matrix1)
print(qr(matrix_h))
