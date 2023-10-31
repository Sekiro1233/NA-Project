from math import e, pi, pow, sqrt


i = 0
f1, f2, f3 = 0.0, 0.0, 0.0
a, a0, a1, a2, a3 = 0.0, 0.0, 0.0, 0.0, 0.0
h1, h2, h3 = 0.0, 0.0, 0.0
g, g0, g1, g2, g3 = 0.0, 0.0, 0.0, 0.0, 0.0
x01, x02, x03 = 0.0, 0.0, 0.0
z1, z2, z3, z0 = 0.0, 0.0, 0.0, 0.0
tol, min_val = 0.0, 10.0
x01, x02, x03, tol = map(float, input().split())

while abs(min_val - g1) >= tol:
    i += 1
    f1 = 15 * x01 + pow(x02, 2) - 4 * x03 - 13
    f2 = pow(x01, 2) + 10 * x02 - x03 - 11
    f3 = pow(x02, 3) - 25 * x03 + 22
    g1 = f1 * f1 + f2 * f2 + f3 * f3

    z1 = 2 * (
        2 * x01 * x01 * x01
        + 15 * x02 * x02
        - 2 * x01 * x02
        + 203 * x01
        - 60 * x03
        - 195
    )
    z2 = 2 * (
        10 * x01 * x01
        + 2 * x02 * x02 * x02
        - 3 * pow(x02, 5)
        - 66 * pow(x02, 2)
        + 74 * x02
        + 30 * x01 * x02
        - 10 * x03
        - 110
        + 75 * pow(x02, 2) * x03
    )
    z3 = 2 * (
        -x01 * x01
        - 4 * x02 * x02
        - 25 * pow(x02, 3)
        - 60 * x01
        - 10 * x02
        + 643 * x03
        - 487
    )
    z0 = sqrt(pow(z1, 2) + pow(z2, 2) + pow(z3, 2))

    if z0 == 0:
        print(
            "Failed, z0 = 0, n = ",
            i,
            ", x1 = ",
            format(x01, ".8f"),
            ", x2 = ",
            format(x02, ".8f"),
            ", x3 = ",
            format(x03, ".8f"),
        )
        exit(0)

    z1 /= z0
    z2 /= z0
    z3 /= z0
    a3 = 1
    g3 = (
        pow(
            (15 * (x01 - a3 * z1) + pow(x02 - a3 * z2, 2) - 4 * (x03 - a3 * z3) - 13), 2
        )
        + pow((pow(x01 - a3 * z1, 2) + 10 * (x02 - a3 * z2) - (x03 - a3 * z3) - 11), 2)
        + pow((pow(x02 - a3 * z2, 3) - 25 * (x03 - a3 * z3) + 22), 2)
    )

    while g3 >= g1:
        a3 /= 2
        g3 = (
            pow(
                (
                    15 * (x01 - a3 * z1)
                    + pow(x02 - a3 * z2, 2)
                    - 4 * (x03 - a3 * z3)
                    - 13
                ),
                2,
            )
            + pow(
                (pow(x01 - a3 * z1, 2) + 10 * (x02 - a3 * z2) - (x03 - a3 * z3) - 11), 2
            )
            + pow((pow(x02 - a3 * z2, 3) - 25 * (x03 - a3 * z3) + 22), 2)
        )

        if a3 < 0.01 * tol:
            print(
                "Failed, a3 = ",
                format(a3, ".8f"),
                ", n = ",
                i,
                ", x1 = ",
                format(x01, ".8f"),
                ", x2 = ",
                format(x02, ".8f"),
                ", x3 = ",
                format(x03, ".8f"),
            )
            exit(0)

    a2 = a3 / 2
    g2 = (
        pow(
            (15 * (x01 - a2 * z1) + pow(x02 - a2 * z2, 2) - 4 * (x03 - a2 * z3) - 13), 2
        )
        + pow((pow(x01 - a2 * z1, 2) + 10 * (x02 - a2 * z2) - (x03 - a2 * z3) - 11), 2)
        + pow((pow(x02 - a2 * z2, 3) - 25 * (x03 - a2 * z3) + 22), 2)
    )

    h1 = (g2 - g1) / a2
    h2 = (g3 - g2) / (a3 - a2)
    h3 = (h2 - h1) / a3
    a0 = (a2 - h1 / h3) / 2
    g0 = (
        pow(
            (15 * (x01 - a0 * z1) + pow(x02 - a0 * z2, 2) - 4 * (x03 - a0 * z3) - 13), 2
        )
        + pow((pow(x01 - a0 * z1, 2) + 10 * (x02 - a0 * z2) - (x03 - a0 * z3) - 11), 2)
        + pow((pow(x02 - a0 * z2, 3) - 25 * (x03 - a0 * z3) + 22), 2)
    )

    min_val = g0
    if g3 < min_val:
        min_val = g3

    if min_val == g0:
        a = a0
    else:
        a = a3

    x01 -= a * z1
    x02 -= a * z2
    x03 -= a * z3
    print(
        "n =",
        i,
        ", x1 =",
        format(x01, ".8f"),
        ", x2 =",
        format(x02, ".8f"),
        ", x3 =",
        format(x03, ".8f"),
        ", g =",
        format(min_val, ".8f"),
    )
