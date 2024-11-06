import numpy as np
import random

from lib.ga_basic import *
from lib.eval_convergence_metric import ConvergenceMetricCalculator
from lib.eval_diversity_metric import DiversityMetricCalculator


# 目标函数
def f1(x1, x2):
    return x1 ** 2 + x2 ** 2


def f2(x1, x2):
    return (x1 - 2) ** 2 + x2 ** 2


# 自变量范围和分辨率
variable_ranges = [(-2, 2), (-2, 2)]
resolution = 500
precision = 0.01  # 期望的搜索精度
num_bits = [8, 8]  # 每个变量的二进制位数
population_size = 80  # 种群大小
archive_size = 20  # 外部种群大小

total_population_size = population_size + archive_size
max_evaluations = 25000  # 最大函数评估次数


# 计算强度和拥挤度
def calculate_strength_and_crowding_distance(population, archive, funcs, variable_ranges, num_bits):
    # 计算每个个体的强度和拥挤度，返回排序后的个体
    strength = {}
    crowding_distance = {}

    # 将种群和外部存档合并
    all_individuals = population + archive
    all_objectives = [calculate_objectives(ind, funcs, variable_ranges, num_bits) for ind in all_individuals]

    # 计算强度（支配关系）
    for i, ind1 in enumerate(all_individuals):
        dominated = 0
        for j, ind2 in enumerate(all_individuals):
            if i != j:
                if (all_objectives[i][0] <= all_objectives[j][0] and all_objectives[i][1] <= all_objectives[j][
                    1]) and not (
                        all_objectives[j][0] <= all_objectives[i][0] and all_objectives[j][1] <= all_objectives[i][1]):
                    dominated += 1
        strength[ind1] = dominated

    # 拥挤度计算
    crowding_distance = {ind: 0 for ind in all_individuals}
    for i in range(2):  # 对每个目标进行拥挤度计算
        sorted_inds = sorted(all_individuals, key=lambda ind: all_objectives[all_individuals.index(ind)][i])
        crowding_distance[sorted_inds[0]] = float('inf')
        crowding_distance[sorted_inds[-1]] = float('inf')
        for j in range(1, len(sorted_inds) - 1):
            crowding_distance[sorted_inds[j]] += (all_objectives[all_individuals.index(sorted_inds[j + 1])][i] -
                                                  all_objectives[all_individuals.index(sorted_inds[j - 1])][i])

    return strength, crowding_distance


# 选择父母
def select_parents(population, strength, crowding_distance):
    # 根据强度和拥挤度选择父母
    selected_parents = sorted(population, key=lambda ind: (strength[ind], -crowding_distance[ind]))
    return selected_parents[:population_size]  # 选择前population_size个个体


# 执行SPEA算法
def spea_algorithm():
    # 初始化种群和外部存档
    num_bits = [calculate_num_bits(min_val, max_val, precision) for (min_val, max_val) in variable_ranges]
    population = initialize_population(population_size, num_bits, variable_ranges)
    archive = []

    evaluations = 0
    while evaluations < max_evaluations:
        # 计算强度和拥挤度
        strength, crowding_distance = calculate_strength_and_crowding_distance(population, archive, [f1, f2],
                                                                               variable_ranges, num_bits)

        # 选择父母
        parents = select_parents(population, strength, crowding_distance)

        # 执行交叉和变异
        offspring = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            off1, off2 = crossover(parent1, parent2)
            offspring.append(mutate(off1))
            offspring.append(mutate(off2))

        # 更新种群
        population = offspring

        # 更新外部存档
        update_archive(population, archive, strength, crowding_distance)

        evaluations += population_size  # 更新评估次数
        # 解码最终存档并返回
        decoded_population = []
        for individual in archive:
            decoded_individual = [
                binary_decode(individual[i:i + num_bits[j]], variable_ranges[j][0], variable_ranges[j][1], num_bits[j])
                for i, j in enumerate(range(len(num_bits)))]
            decoded_population.append(decoded_individual)

        return decoded_population  # 返回解码后的种群


# 更新存档
def update_archive(population, archive, strength, crowding_distance):
    # 将种群与外部种群合并，并按强度和拥挤度选择更新存档
    combined_population = population + archive
    strength, crowding_distance = calculate_strength_and_crowding_distance(population, archive, [f1, f2],
                                                                           variable_ranges, num_bits)
    sorted_population = sorted(combined_population, key=lambda ind: (strength[ind], -crowding_distance[ind]))
    archive[:] = sorted_population[:archive_size]  # 保持存档大小不变


# 生成理论帕累托最优解集
def generate_theoretical_pareto_front():
    # 对于 f1 和 f2，理论帕累托前沿是 x1 = 1 的线
    x2_values = np.linspace(-2, 2, 500)
    theoretical_front = np.array([[np.ones(x2_values.shape), x2_values]])
    return theoretical_front


# 运行算法并输出结果
optimal_front = generate_theoretical_pareto_front()
final_archive = spea_algorithm()
print(final_archive)
caculator1 = DiversityMetricCalculator(final_archive)
caculator2 = ConvergenceMetricCalculator(optimal_front, final_archive, 500)
metric1 = caculator1.get_diversity_metric()
metric2 = caculator2.calculate_convergence_metric()
print(metric1)
print(metric2)
