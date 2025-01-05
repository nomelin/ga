import numpy as np


# POL problem
def f1(x1, x2, t):
    A1 = 0.5 * np.sin(1) - a(t) * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
    A2 = 1.5 * np.sin(1) - np.cos(1) + a(t) * np.sin(2) - 0.5 * np.cos(2)
    B1 = 0.5 * np.sin(x1) - a(t) * np.cos(x1) + np.sin(x2) - 1.5 * np.cos(x2)
    B2 = 1.5 * np.sin(x1) - np.cos(x1) + a(t) * np.sin(x2) - 0.5 * np.cos(x2)
    return 1 + (A1 - B1) + (A2 - B2)


def f2(x1, x2, t):
    return (x1 + 3) + (x2 + 1) + a(t)


def a(t):
    return t + 1


# 定义变量范围
variable_range = [(-np.pi, np.pi), (-np.pi, np.pi)]

# 定义目标函数
funcs = [f1, f2]
is_dynamic = True
