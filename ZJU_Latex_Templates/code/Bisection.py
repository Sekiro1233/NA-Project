from math import pow

e = 2.718281828459
n = 0
a, b, exp = map(float, input().split())
fa = pow(e, a) - a * a + 3 * a - 2
while b - a > exp:
    mid = (a + b) / 2
    fmid = pow(e, mid) - mid * mid + 3 * mid - 2
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
