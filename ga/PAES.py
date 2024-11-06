import numpy as np
from lib.eval_convergence_metric import ConvergenceMetricCalculator
from lib.eval_diversity_metric import DiversityMetricCalculator
from ga.nsgaii import dominates
from lib.ga_basic import *


def dominance_check(individual1, individual2, funcs, variable_ranges, num_bits):
    """
    判断个体1是否支配个体2（即，个体1在所有目标上表现不差于个体2，且至少在一个目标上表现更好）。
    """
    objectives1 = calculate_objectives(individual1, funcs, variable_ranges, num_bits)
    objectives2 = calculate_objectives(individual2, funcs, variable_ranges, num_bits)

    dominate1 = all(o1 <= o2 for o1, o2 in zip(objectives1, objectives2))  # 个体1是否不差于个体2
    dominate2 = all(o2 <= o1 for o1, o2 in zip(objectives1, objectives2))  # 个体2是否不差于个体1

    return dominate1, dominate2


def update_archive(archive, individual, funcs, variable_ranges, num_bits, archive_size):
    """
    更新存档，存储非支配解。
    """
    # 临时存档
    new_archive = archive + [individual]

    # 非支配排序
    fronts = fast_non_dominated_sort(new_archive, funcs, variable_ranges, num_bits)

    # 从前沿中选择最优个体
    archive = []
    for front in fronts:
        if len(archive) + len(front) > archive_size:
            # 若加入当前前沿后存档超过最大存档大小，进行拥挤度排序
            sorted_front = sort_by_crowding_distance(front, funcs, variable_ranges, num_bits)
            archive.extend(sorted_front[:archive_size - len(archive)])
            break
        else:
            archive.extend(front)

    return archive


def fast_non_dominated_sort(population, funcs, variable_ranges, num_bits):
    """
    快速非支配排序：对种群进行非支配排序，返回所有个体的前沿集合。
    """
    fronts = []
    domination_count = [0] * len(population)
    dominated_solutions = [[] for _ in range(len(population))]

    for p in range(len(population)):
        for q in range(len(population)):
            if p != q:
                dominate1, dominate2 = dominance_check(population[p], population[q], funcs, variable_ranges, num_bits)
                if dominate1:
                    dominated_solutions[p].append(q)
                elif dominate2:
                    domination_count[p] += 1

        if domination_count[p] == 0:
            if not fronts:
                fronts.append([p])
            else:
                fronts[0].append(p)

    # 从支配关系中获得非支配前沿
    i = 0
    while i < len(fronts):
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)

    return fronts


def sort_by_crowding_distance(front, funcs, variable_ranges, num_bits):
    """
    按照拥挤度对个体进行排序。拥挤度较高的个体优先。
    """
    # 计算每个个体的拥挤度
    distances = [0] * len(front)
    for i in range(len(funcs)):
        sorted_front = sorted(front, key=lambda ind: calculate_objectives(ind, funcs, variable_ranges, num_bits)[i])
        distances[0] = distances[-1] = float('inf')
        for i in range(1, len(front) - 1):
            distances[i] += (calculate_objectives(sorted_front[i + 1], funcs, variable_ranges, num_bits)[i] -
                             calculate_objectives(sorted_front[i - 1], funcs, variable_ranges, num_bits)[i])

    # 根据拥挤度值对个体进行排序
    front_sorted = [x for _, x in sorted(zip(distances, front), reverse=True)]
    return front_sorted

def paes(pop_size, num_bits, variable_ranges, funcs, archive_size, max_generations, crossover_rate=0.9,
         mutation_rate=0.01):
    """
    PAES算法的主循环。
    """
    archive = []

    # 初始化种群
    population = initialize_population(pop_size, num_bits, variable_ranges)

    # 评估种群
    for individual in population:
        archive = update_archive(archive, individual, funcs, variable_ranges, num_bits, archive_size)

    # 进行25000次迭代
    for generation in range(max_generations):
        new_population = []
        for _ in range(pop_size):
            # 从存档中选择两个父代
            parent1 = random.choice(archive)
            parent2 = random.choice(archive)

            # 执行交叉操作
            offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)

            # 执行突变操作
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)

            # 评估后代
            new_population.extend([offspring1, offspring2])

        # 将新生成的个体加入存档
        for individual in new_population:
            archive = update_archive(archive, individual, funcs, variable_ranges, num_bits, archive_size)

    print(f"Generation {generation}: Archive Size = {len(archive)}")


    return archive


def print_population_values(archive, variable_ranges, num_bits):
    """
    输出存档中所有个体的实数值。

    参数:
        archive (list of str): 存档中的所有个体（以二进制字符串表示）。
        variable_ranges (list of tuples): 每个变量的范围 [(var_min, var_max), ...]。
        num_bits (list of int): 每个变量的二进制位数。
    """
    for i, individual in enumerate(archive):
        decoded_values = decode_individual(individual, variable_ranges, num_bits)
        print(f"Individual {i + 1}: {decoded_values}")


# ===========================
# 目标函数 f1 和 f2
# ===========================
def f1(x1, x2):
    return x1 ** 2 + x2 ** 2


def f2(x1, x2):
    return (x1 - 2) ** 2 + x2 ** 2


# ===========================
# 参数设置
# ===========================
variable_ranges = [(-2, 2), (-2, 2)]
resolution = 500  # 离散的取值数量
precision = 0.01  # 期望的搜索精度
num_bits = [8, 8]  # 每个变量的二进制位数
archive_size = 100  # 存档大小
max_generations = 25000  # 最大迭代次数
pop_size = 100 # 种群大小

# ===========================
# 运行PAES算法
# ===========================
archive = paes(pop_size, num_bits, variable_ranges, [f1, f2], archive_size, max_generations)
# 输出最后种群的所有个体值
print("Final population values:")
print_population_values(archive, variable_ranges, num_bits)