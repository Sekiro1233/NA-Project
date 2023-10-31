from math import cos

n = 0
a, b, exp = map(float, input().split())
fa = a * cos(a) - 2 * a * a + 3 * a - 1
while b - a > exp:
    mid = (a + b) / 2
    fmid = mid * cos(mid) - 2 * mid * mid + 3 * mid - 1
    n = n + 1
    if fmid == 0:
        print("n = {}, solution = {:.8f}".format(n, mid))
        break
    if fmid * fa > 0:
        a = mid
    else:
        b = mid
    print(
        "n = {}, mid = {:.8f}, a-b = {:.8f}, f(mid) = {:.8f}".format(
            n, mid, a - b, fmid
        )
    )
