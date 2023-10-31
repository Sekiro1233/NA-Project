import numpy as np

n = int(input())
a = np.zeros((n, n + 1))
s = np.zeros(n)
row = np.zeros(n, dtype=int)

for i in range(n):
    a[i] = [float(x) for x in input().split()]
    s[i] = max(np.abs(a[i]))
    row[i] = i

for i in range(n - 1):
    ratios = [np.abs(a[row[j]][i]) / s[row[j]] for j in range(i, n)]
    p = ratios.index(max(ratios)) + i
    if a[row[p]][i] == 0:
        print("Algorithm failed")
        exit(0)
    if row[i] != row[p]:
        row[i], row[p] = row[p], row[i]
    for j in range(i + 1, n):
        m = a[row[j]][i] / a[row[i]][i]
        a[row[j]] = a[row[j]] - m * a[row[i]]

if a[row[n - 1]][n - 1] == 0:
    print("Algorithm failed")
    exit(0)

x = np.zeros(n)
x[n - 1] = a[row[n - 1]][n] / a[row[n - 1]][n - 1]

for i in range(n - 2, -1, -1):
    x[i] = (a[row[i]][n] - sum(a[row[i]][j] * x[j] for j in range(i + 1, n))) / a[
        row[i]
    ][i]

for i in range(n):
    print("x[{}]={:.8f}".format(i + 1, x[i]))
