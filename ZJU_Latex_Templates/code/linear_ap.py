import numpy as np
import matplotlib.pyplot as plt

# input data
x = np.array([0, 2, 4, 5])
y = np.array([6, 8, 14, 20])

# compute linear least squares polynomial approximation
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x**2)

a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
a0 = (sum_y - a1 * sum_x) / n

y_fit = a1 * x + a0
E = np.sum((y - y_fit) ** 2)

print("a1:", a1)
print("a0:", a0)
print("Error of the approximation:", E)

plt.scatter(x, y, label="Data")
plt.plot(x, y_fit, label="Linear Approximation")
plt.legend()
plt.show()
