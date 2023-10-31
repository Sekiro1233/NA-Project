from math import fabs, sqrt, pow, e


def g(x):
    return x - (3 * x * x - pow(e, x)) / (6 * x - pow(e, x))


n, t = 0, 1

p0, exp = map(float, input().split())
while t > exp:
    n = n + 1
    p = g(p0)
    t = fabs(p - p0)
    print("n = {}, p_n = {:.8f}, |p_n-g(p_n)| = {:.8f}".format(n, p, fabs(p - p0)))
    p0 = p
