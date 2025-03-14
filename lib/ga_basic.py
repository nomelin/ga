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
        num_bits (list of int): 每个变量的二进制位数。
        variable_ranges (list of tuples): 每个变量的范围 [(var_min, var_max), ...]。

    返回:
        list of str: 包含每个个体二进制编码的种群。

    示例:
        >> initialize_population(3, [8, 3], [(-2, 2), (-2, 2)])
        ['10000101101', '11100111011', '01010010001']
    """
    population = []
    for _ in range(pop_size):
        individual = "".join(
            binary_encode(
                random.uniform(var_min, var_max), var_min, var_max, bits
            ) for (var_min, var_max), bits in zip(variable_ranges, num_bits)
        )
        population.append(individual)
    return population


# ======================
# 遗传算法基本算子
# ======================


def encode_individual(individual, variable_ranges, precision):
    """
    将个体中的所有浮点数值编码为二进制字符串并拼接。

    参数:
        individual (list of float): 要编码的个体列表，每个元素是浮点数。
        variable_ranges (list of tuple): 变量的范围列表，每个元素是一个二元组，表示最小值和最大值。
        precision (float): 期望的搜索精度（决定二进制位数）。

    返回:
        str: 拼接后的二进制字符串。
    """
    binary_string = ''

    # 遍历每个个体的维度及其对应的变量范围
    for i, value in enumerate(individual):
        var_min, var_max = variable_ranges[i]  # 对应维度的最小值和最大值
        num_bits = calculate_num_bits(var_min, var_max, precision)  # 计算当前维度需要的二进制位数
        binary_string += binary_encode(value, var_min, var_max, num_bits)  # 编码当前个体的值并添加到二进制字符串

    return binary_string


def decode_individual(individual, variable_ranges, num_bits):
    """
    解码个体的二进制串为对应的实数列表。

    参数:
        individual (str): 个体的二进制字符串。
        variable_ranges (list of tuples): 每个变量的范围 [(var_min, var_max), ...]。
        num_bits (list of int): 每个变量的二进制位数。

    返回:
        list of float: 解码后的实数列表。

    示例:
        >> decode_individual('11001101100', [(-2, 2), (-2, 2)], [8, 3])
        [1.215686274509804, 0.2857142857142856]
    """
    decoded = []
    start_index = 0
    for (var_min, var_max), bits in zip(variable_ranges, num_bits):
        bin_str = individual[start_index: start_index + bits]
        decoded.append(binary_decode(bin_str, var_min, var_max, bits))
        start_index += bits
    return decoded


def encode_Individual(decoded_vars, variable_ranges, num_bits):
    """
    将解码后的实数列表编码为二进制串。
    参数:
        decoded_vars (list of float): 解码后的实数列表。
        variable_ranges (list of tuples): 每个变量的范围 [(var_min, var_max), ...]。
        num_bits (list of int): 每个变量的二进制位数。

    返回:
        str: 编码后的二进制串。

    示例:
        >> encode_Individual([1.215686274509804, 0.2857142857142856], [(-2, 2), (-2, 2)], [8, 3])
        '11001101100'
    """
    # print(f"decoded_vars: {decoded_vars}, variable_ranges: {variable_ranges}, num_bits: {num_bits}")
    binary_string = ''.join(
        binary_encode(var, var_min, var_max, bits)
        for var, (var_min, var_max), bits in zip(decoded_vars, variable_ranges, num_bits)
    )
    return binary_string


def calculate_objectives(individual, funcs, variable_ranges, num_bits, t):
    """
    计算个体的目标函数值，支持时间变量 t。

    参数:
        individual (str): 个体的二进制字符串。
        funcs (list of functions): 目标函数列表。
        variable_ranges (list of tuples): 每个变量的范围 [(var_min, var_max), ...]。
        num_bits (list of int): 每个变量的二进制位数。
        t (float): 时间变量。

    返回:
        list of float: 个体的目标函数值列表。

    示例:
        >> calculate_objectives('1010101010101010', [f1, f2], [(-2, 2), (-2, 2)], [8, 8], 0.5)
        [1.0, 0.5]
    """
    decoded_vars = decode_individual(individual, variable_ranges, num_bits)
    if t is not None:
        return [func(*decoded_vars, t) for func in funcs]
    else:
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


if __name__ == '__main__':
    a = encode_Individual([1.215686274509804, 0.2857142857142856], [(-10, 10), (-2, 2)], [15, 3])
    print(a)
    b = decode_individual(a, [(-5, 2), (-2, 2)], [8, 3])
    print(b)
