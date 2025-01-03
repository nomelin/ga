import numpy as np


# FON problem

def f1(x1, x2, x3):
    return 1 - np.exp(-((x1 - 1 / np.sqrt(3)) ** 2 + (x2 - 1 / np.sqrt(3)) ** 2 + (x3 - 1 / np.sqrt(3)) ** 2))


def f2(x1, x2, x3):
    return 1 - np.exp(-((x1 + 1 / np.sqrt(3)) ** 2 + (x2 + 1 / np.sqrt(3)) ** 2 + (x3 + 1 / np.sqrt(3)) ** 2))


variable_ranges = [(-4, 4), (-4, 4), (-4, 4)]
funcs = [f1, f2]
