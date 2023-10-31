import numpy as np

n = int(input())
a = np.zeros((n, n))
b = np.zeros(n)
x0 = np.zeros(n)
x = np.zeros(n)

for i in range(n):
    a[i] = list(map(float, input().split()))

b = list(map(float, input().split()))

x0 = list(map(float, input().split()))

k = 0
while k < 3:
    for i in range(n):
        sum = 0
        for j in range(n):
            if j != i:
                sum += a[i][j] * x0[j]
        x[i] = (b[i] - sum) / a[i][i]

    x0 = x
    k += 1
    print("n = {}, x = ".format(k), end="")
    for xi in x:
        print("{:.8f}".format(xi))
