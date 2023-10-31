from math import fabs, sqrt, pow, e


def g(x):
    return sqrt(pow(e, p0) / 3)


n, t = 0, 1

p0, exp = map(float, input().split())
while t > exp:
    n = n + 1
    p = g(p0)
    t = fabs(p - p0)
    print("n = {}, p_n = {:.8f}, |p_n-g(p_n)| = {:.8f}".format(n, p, fabs(p - p0)))
    p0 = p
