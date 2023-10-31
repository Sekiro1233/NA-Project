from math import cos, sin

i = 0
p0, p = 1, 1
p0 = int(input())
while i <= 1:
    i += 1
    p = p0 - (-(p0**3) - cos(p0)) / (-3 * p0**2 + sin(p0))
    print("n = {}, p = {:.8f}, |p0-p| = {:.8f}".format(i, p, abs(p0 - p)))
    p0 = p
