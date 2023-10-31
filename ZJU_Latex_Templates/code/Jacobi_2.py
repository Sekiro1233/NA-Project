import numpy as np
n = int(input())
a = np.zeros((n, n))
b = np.zeros(n)
x0 = np.zeros(n)
x = np.zeros(n)
norm = np.zeros(n)
tol = float(input())
for i in range(n):
    a[i] = list(map(float, input().split()))
b = list(map(float, input().split()))
x0 = list(map(float, input().split()))
k = 1
while True:
    for i in range(n):
        sum = 0
        for j in range(n):
            if j != i:
                sum += a[i][j] * x0[j]
        x[i] = (b[i] - sum) / a[i][i]
        norm[i] = abs(x[i] - x0[i])
    for i in range(n):
        x0[i] = x[i]
    k += 1
    print(f"n = {k-1}, x = ", end="")
    for xi in x:
        print("{:.8f}".format(xi))
    norml = max(norm)
    if norml <= tol:
        break