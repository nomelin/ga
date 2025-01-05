from math import sin


def f1(x1, x2, t):
    return x1 ** 3 + 2 * x2 ** 2 + 3 * x1 * x2 * a(t) + a(t)


def f2(x1, x2, t):
    return (x1 - a(t)) ** 2 + 4 * x2 ** 2 - 2 * x1 * x2


def a(t):
    return sin(t) + 2


variable_range = [(-2, 2), (-2, 2)]
funcs = [f1, f2]
is_dynamic = True
