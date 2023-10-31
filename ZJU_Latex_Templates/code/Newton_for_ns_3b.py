from numpy import array, linalg

i = 0
x01, x02, x03 = map(float, input().split())
while i <= 1:
    i += 1
    f1 = x01**2 + x02 - 37
    f2 = x01 - x02**2 - 5
    f3 = x01 + x02 + x03 - 3
    jz = array(
        [
            [2 * x01, 1, 0],
            [1, -2 * x02, 0],
            [1, 1, 1],
        ]
    )
    jnz = linalg.inv(jz)
    y1 = jnz[0][0] * f1 + jnz[0][1] * f2 + jnz[0][2] * f3
    y2 = jnz[1][0] * f1 + jnz[1][1] * f2 + jnz[1][2] * f3
    y3 = jnz[2][0] * f1 + jnz[2][1] * f2 + jnz[2][2] * f3
    x1 = x01 - y1
    x2 = x02 - y2
    x3 = x03 - y3
    print("n = {}, x1 = {:.8f}, x2 = {:.8f}, x3 = {:.8f}".format(i, x1, x2, x3))
    x01, x02, x03 = x1, x2, x3
