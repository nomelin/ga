def f1(x1, x2, t):
    return x1 ** 2 + x2 ** 2 + a(t)


def f2(x1, x2, t):
    return (x1 - a(t)) ** 2 + x2 ** 2


def a(t):
    return t + 1


variable_range = [(-2, 2), (-2, 2)]
funcs = [f1, f2]
is_dynamic = True
