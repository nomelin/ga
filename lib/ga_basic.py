import random

import numpy as np


# ======================
# 基本函数
# ======================

def binary_encode(value, var_min, var_max, num_bits):
    """
    将实数编码为指定精度的二进制字符串。

    参数:
        value (float): 要编码的实数。
        var_min (float): 变量的最小值。
        var_max (float): 变量的最大值。
        num_bits (int): 二进制位数。

    返回:
        str: 二进制字符串表示的编码值。

    示例:
        >> binary_encode(1.5, 0, 2, 8)
        '11000000'
    """
    scale = (value - var_min) / (var_max - var_min)
    int_value = int(scale * (2 ** num_bits - 1))
    return np.binary_repr(int_value, width=num_bits)


def binary_decode(bin_str, var_min, var_max, num_bits):
    """
    将二进制字符串解码为实数值。

    参数:
        bin_str (str): 要解码的二进制字符串。
        var_min (float): 变量的最小值。
        var_max (float): 变量的最大值。
        num_bits (int): 二进制位数。

    返回:
        float: 解码后的实数值。

    示例:
        >> binary_decode('11000000', 0, 2, 8)
        1.5
    """
    int_value = int(bin_str, 2)
    scale = int_value / (2 ** num_bits - 1)
    return var_min + scale * (var_max - var_min)


def calculate_num_bits(var_min, var_max, precision):
    """
    根据期望的搜索精度计算所需的二进制位数。

    参数:
        var_min (float): 变量的最小值。
        var_max (float): 变量的最大值。
        precision (float): 期望的搜索精度（例如 0.01 表示精确到小数点后两位）。

    返回:
        int: 表示该精度所需的二进制位数。

    示例:
        >> calculate_num_bits(-5, 5, 0.01)
        10
    """
    range_size = var_max - var_min
    num_bits = int(np.ceil(np.log2(range_size / precision)))
    return num_bits


def initialize_population(pop_size, num_bits, variable_ranges):
    """
    初始化种群并编码为二进制个体。

    参数:
        pop_size (int): 种群大小。
        num_bits (int): 每个变量的二进制位数。
        variable_ranges (list of tuples): 每个变量的范围 [(var_min, var_max), ...]。需要几个变量就有几个范围。

    返回:
        list of str: 包含每个个体二进制编码的种群。

    示例:
        >> initialize_population(3, 8, [(-2, 2), (-2, 2)])
        ['1011100101101111', '0001011001011001', '1011110110101111']

        >> initialize_population(3, 8, [(-2, 2)])
        ['10110000', '00011011', '01101100']
    """
    population = []
    for _ in range(pop_size):
        individual = "".join(
            binary_encode(
                random.uniform(var_min, var_max), var_min, var_max, num_bits
            ) for var_min, var_max in variable_ranges
        )
        population.append(individual)
    return population


# ======================
# 遗传算法基本算子
# ======================

def decode_individual(individual, variable_ranges, num_bits):
    """
    解码个体的二进制串为对应的实数列表。

    参数:
        individual (str): 个体的二进制字符串。
        variable_ranges (list of tuples): 每个变量的范围 [(var_min, var_max), ...]。需要几个变量就有几个范围。
        num_bits (int): 每个变量的二进制位数。

    返回:
        list of float: 解码后的实数列表。

    示例:
        >> decode_individual('1010101010101010', [(-2, 2), (-2, 2)], 8)
        [0.0, 1.25]
    """
    decoded = []
    for i, (var_min, var_max) in enumerate(variable_ranges):
        bin_str = individual[i * num_bits: (i + 1) * num_bits]
        decoded.append(binary_decode(bin_str, var_min, var_max, num_bits))
    return decoded


def calculate_objectives(individual, funcs, variable_ranges, num_bits):
    """
    计算个体的目标函数值。

    参数:
        individual (str): 个体的二进制字符串。
        funcs (list of functions): 目标函数列表。
        variable_ranges (list of tuples): 每个变量的范围 [(var_min, var_max), ...]。需要几个变量就有几个范围。
        num_bits (int): 每个变量的二进制位数。

    返回:
        list of float: 个体的目标函数值列表。

    示例:
        >> calculate_objectives('1010101010101010', [f1, f2], [(-2, 2), (-2, 2)], 8)
        [1.25, 0.75]
    """
    decoded_vars = decode_individual(individual, variable_ranges, num_bits)
    return [func(*decoded_vars) for func in funcs]


def crossover(parent1, parent2, crossover_rate=0.9):
    """
    执行单点交叉操作。交叉点为随机位置。

    参数:
        parent1 (str): 父代个体1的二进制字符串。
        parent2 (str): 父代个体2的二进制字符串。
        crossover_rate (float): 交叉概率，默认值为0.9。

    返回:
        tuple of str: 两个子代个体的二进制字符串。

    示例:
        >> crossover('10101010', '01010101')
        ('10110101', '01001010')
    """
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:point] + parent2[point:]
        offspring2 = parent2[:point] + parent1[point:]
        return offspring1, offspring2
    return parent1, parent2


def mutate(individual, mutation_rate=0.01):
    """
    执行突变操作。

    参数:
        individual (str): 个体的二进制字符串。
        mutation_rate (float): 突变概率，默认值为0.01。

    返回:
        str: 突变后的二进制字符串。

    示例:
        >> mutate('10101010', 0.1)
        '10001010'
    """
    mutated = "".join(
        str(1 - int(bit)) if random.random() < mutation_rate else bit
        for bit in individual
    )
    return mutated
