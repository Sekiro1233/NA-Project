from math import fabs, sqrt, pow, e


def g1(x1, x2):
    return (x1**2 + x2**2 + 8) / 10


def g2(x1, x2):
    return (x1 * x2**2 + x1 + 8) / 10


i = 0

x01, x02 = map(float, input().split())
while i <= 1:
    i += 1
    x_1, x_2 = g1(x01, x02), g2(x01, x02)
    print("n = {}, x1 = {:.8f}, x2 = {:.8f}, ".format(i, x_1, x_2))
    x01, x02 = x_1, x_2
