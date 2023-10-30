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


print(
    householder(np.array([[4, 1, -2, 2], [1, 2, 0, 1], [-2, 0, 3, -2], [2, 1, -2, -1]]))
)
