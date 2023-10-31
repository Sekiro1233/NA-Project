import numpy as np
from scipy.sparse import random
from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix


# 题目1：生成随机矩阵******************************************************************************
print("题目1: ————————————————————————————————————————")

# (1)生成一个10x10的随机矩阵
matrix1 = np.random.rand(10, 10) * 10
print("(1)")
print("矩阵(1): ")

print(matrix1)

# (2)生成一个10000x10000维度且密度为0.001的随机稀疏矩阵，并计算非零元素的数量
matrix_2 = random(10000, 10000, density=0.001) * 10

nonzero_cnt = matrix_2.count_nonzero()
# 首先将其转换为稀疏矩阵格式节省内存
matrix2 = csc_matrix(matrix_2)


print("(2)")
print("非零元素数量: ", nonzero_cnt)

# (3)计算matrix1和matrix2的特征值
eigenvalues1 = np.linalg.eigvals(matrix1)
eigenvalues2, _ = eigs(matrix2, k=8)
print("(3)")
print("矩阵(1)特征值: ")
print(eigenvalues1)
print("矩阵(2)特征值: ")
print(eigenvalues2)

# 题目2：给出 Power Method 的伪代码并⽤代码实现，能够输出绝对值最⼤的特征值********************************************************************************
print("题目2: ————————————————————————————————————————")


# 实现power method
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
eigenvalue_pm2, _ = power_method(matrix2.toarray(), 1e-6, 1000)
print("(2)")
print(
    "矩阵(2)绝对值最大的特征值: ", eigenvalue_pm2
)  # matrix2是一个很大的稀疏矩阵，而power method算法在处理稀疏矩阵时可能会出现问题，因此这里的结果可能不准确

# 题目3：给出 QR 算法的伪代码并⽤代码实现，并能够实现输出前k个绝对值最⼤的特征值，其中k为⾃定义参数**************************************
print("题目3: ————————————————————————————————————————")


# 实现QR分解
def givens_reduce(matrix):
    n = matrix.shape[0]
    R = matrix
    G_list = []

    for j in range(n):
        for i in range(j + 1, n):
            if R[i][j] == 0:
                continue

            a = R[j][j]
            b = R[i][j]

            base = np.sqrt(np.power(a, 2) + np.power(b, 2))
            c = np.true_divide(a, base)
            s = np.true_divide(b, base)

            G = np.identity(n)
            G[j][j] = c
            G[i][j] = -s
            G[j][i] = s
            G[i][i] = c

            R = np.matmul(G, R)

            G_list.append(G)

    R = R[0:n]

    Q_prime = np.identity(n)
    for G in G_list:
        Q_prime = np.matmul(G, Q_prime)

    Q = np.transpose(Q_prime[0:n])

    return Q, R


# 实现QR迭代求解特征值
def qr_iteration(matrix, k):
    n = matrix.shape[0]
    for i in range(1000):
        q, r = givens_reduce(matrix)
        matrix = np.dot(r, q)
    matrix_1 = np.dot(q, r)
    eigenvalues = []
    for i in range(n):
        eigenvalues.append(matrix_1[i, i])
        eigenvalues = sorted(eigenvalues, key=lambda x: abs(x), reverse=True)[:k]
    return eigenvalues


print("(1)")
print("矩阵(1)前4个绝对值最大的特征值为：")
print(qr_iteration(matrix1, 4))
print("(2)")
print("矩阵(2)前5个绝对值最大的特征值为：")  # print(qr_iteration(matrix2.toarray(), 5))

# 题目4：⽤代码实现 Arnoldi 迭代算法，并能够实现输出前 k 个绝对值最⼤的特征值，其中 k 为自定义参数**************************************
print("题目4: ————————————————————————————————————————")


# 对于小的方阵，我们直接将其全部计算，得到的特征值更准确
def arnoldi_iteration1(matrix, k):
    n = matrix.shape[0]
    Q = np.zeros((n, n + 1))
    H = np.zeros((n + 1, n))
    b = np.random.rand(n)
    Q[:, 0] = b / np.linalg.norm(b)
    for j in range(n):
        v = np.dot(matrix, Q[:, j])
        for i in range(j + 1):
            H[i, j] = np.dot(Q[:, i], v)
            v = v - H[i, j] * Q[:, i]
        H[j + 1, j] = np.linalg.norm(v)
        if H[j + 1, j] == 0:
            break
        Q[:, j + 1] = v / H[j + 1, j]
    eigenvalues = qr_iteration(H[:n,], k)
    return eigenvalues


# 对于大的稀疏矩阵，我们只考虑前k列(k<<n)，计算部分特征值，得到的特征值可能不准确
def arnoldi_iteration2(matrix, k):
    n = matrix.shape[0]
    Q = np.zeros((n, k + 1))
    H = np.zeros((k + 1, k))
    b = np.random.rand(n)
    Q[:, 0] = b / np.linalg.norm(b)
    for j in range(k):
        v = np.dot(matrix, Q[:, j])
        for i in range(j + 1):
            H[i, j] = np.dot(Q[:, i], v)
            v = v - H[i, j] * Q[:, i]
        H[j + 1, j] = np.linalg.norm(v)
        if H[j + 1, j] == 0:
            break
        Q[:, j + 1] = v / H[j + 1, j]
    eigenvalues = qr_iteration(H[:k,], k)
    return eigenvalues


print("(1)")
print("矩阵(1)前6个绝对值最大的特征值为：")
print(arnoldi_iteration1(matrix1, 6))
print("(2)")
print("矩阵(2)前7个绝对值最大的特征值为：")
print(arnoldi_iteration2(matrix2.toarray(), 7))

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton
from PyQt5.QtGui import QFont


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle("Sparse Matrix Operations")
        self.setGeometry(100, 100, 800, 600)

        # Set font
        font = QFont()
        font.setPointSize(12)
        self.setFont(font)

        # Create text boxes
        self.textbox1 = QTextEdit()
        self.textbox2 = QTextEdit()
        self.textbox3 = QTextEdit()
        self.textbox4 = QTextEdit()

        # Create button
        self.button = QPushButton("Run Code")
        self.button.clicked.connect(self.run_code)

        # Create layout
        vbox = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()

        hbox1.addWidget(self.textbox1)
        hbox1.addWidget(self.textbox2)
        hbox2.addWidget(self.textbox3)
        hbox2.addWidget(self.textbox4)

        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addWidget(self.button)

        self.setLayout(vbox)

    def run_code(self):
        # Run code for question 1
        self.textbox1.setText("题目1: ————————————————————————————————————————\n\n")
        matrix1 = np.random.rand(10, 10) * 10
        matrix_2 = random(10000, 10000, density=0.001) * 10
        nonzero_cnt = matrix_2.count_nonzero()
        matrix2 = csc_matrix(matrix_2)
        eigenvalues1 = np.linalg.eigvals(matrix1)
        eigenvalues2, _ = eigs(matrix2, k=8)
        self.textbox1.append("(1)\n")
        self.textbox1.append("矩阵(1):\n")
        self.textbox1.append(str(matrix1) + "\n\n")
        self.textbox1.append("(2)\n")
        self.textbox1.append("非零元素数量: " + str(nonzero_cnt) + "\n\n")
        self.textbox1.append("(3)\n")
        self.textbox1.append("矩阵(1)特征值:\n")
        self.textbox1.append(str(eigenvalues1) + "\n")
        self.textbox1.append("矩阵(2)特征值:\n")
        self.textbox1.append(str(eigenvalues2) + "\n")

        # Run code for question 2
        self.textbox2.setText("题目2: ————————————————————————————————————————\n\n")
        eigenvalue_pm1, _ = power_method(matrix1, 1e-6, 1000)
        eigenvalue_pm2, _ = power_method(matrix2.toarray(), 1e-6, 1000)
        self.textbox2.append("(1)\n")
        self.textbox2.append("矩阵(1)绝对值最大的特征值: " + str(eigenvalue_pm1) + "\n\n")
        self.textbox2.append("(2)\n")
        self.textbox2.append("矩阵(2)绝对值最大的特征值: " + str(eigenvalue_pm2) + "\n")

        # Run code for question 3
        self.textbox3.setText("题目3: ————————————————————————————————————————\n\n")
        eigenvalues1 = qr_iteration(matrix1, 4) # eigenvalues2 = qr_iteration(matrix2.toarray(), 5)
        self.textbox3.append("(1)\n")
        self.textbox3.append("矩阵(1)前4个绝对值最大的特征值为：\n")
        self.textbox3.append(str(eigenvalues1) + "\n\n")
        self.textbox3.append("(2)\n")
        self.textbox3.append("矩阵(2)前5个绝对值最大的特征值为：\n")
        self.textbox3.append(str(eigenvalues2) + "\n")

        # Run code for question 4
        self.textbox4.setText("题目4: ————————————————————————————————————————\n\n")
        eigenvalues1 = arnoldi_iteration1(matrix1, 6)
        eigenvalues2 = arnoldi_iteration2(matrix2.toarray(), 7)
        self.textbox4.append("(1)\n")
        self.textbox4.append("矩阵(1)前6个绝对值最大的特征值为：\n")
        self.textbox4.append(str(eigenvalues1) + "\n\n")
        self.textbox4.append("(2)\n")
        self.textbox4.append("矩阵(2)前7个绝对值最大的特征值为：\n")
        self.textbox4.append(str(eigenvalues2) + "\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
