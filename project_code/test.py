def householder(a):
    n = a.shape[0]
    if np.allclose(a, a.T):  # 判断是否为对称矩阵
        return a
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
print('Householder变换得到的对三角矩阵为：')
print(householder(matrix1))
# 基于Householder变换实现QR法求解特征值
def qr(matrix, tol=1e-6, max_iter=1000):
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


matrix_h = householder(matrix1)
print(qr(matrix_h))