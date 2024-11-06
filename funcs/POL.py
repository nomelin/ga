import numpy as np


# POL problem
def f1(x1, x2):
    A1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
    A2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)
    B1 = 0.5 * np.sin(x1) - 2 * np.cos(x1) + np.sin(x2) - 1.5 * np.cos(x2)
    B2 = 1.5 * np.sin(x1) - np.cos(x1) + 2 * np.sin(x2) - 0.5 * np.cos(x2)
    return 1 + (A1 - B1) ** 2 + (A2 - B2) ** 2


def f2(x1, x2):
    return (x1 + 3) ** 2 + (x2 + 1) ** 2


# 定义变量范围
variable_ranges = [(-np.pi, np.pi), (-np.pi, np.pi)]

# 定义目标函数
funcs = [f1, f2]
