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


