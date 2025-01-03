import random
import numpy as np

from ga.nsgaii import adapter_initialize_population, adapter_calculate_objectives, adapter_decode_individual, \
    adapter_crossover, adapter_mutate
from lib.eval_convergence_metric import ConvergenceMetricCalculator
from lib.eval_diversity_metric import DiversityMetricCalculator
from lib.ga_basic import *
from lib.visual import ObjectiveVisualizer

# ======================
# SPEA 算法参数
# ======================
variable_ranges = [(-2, 2), (-2, 2)]
precision = 0.01
num_bits = [8, 8]  # 每个变量的二进制位数
population_size = 80
archive_size = 20
total_population_size = population_size + archive_size
max_evaluations = 25000
save_gif = True  # 设置为True以保存GIF
gif_name = 'spea_visualization'

# ======================
# 目标函数
# ======================
def f1(x1, x2):
    return x1 ** 2 + x2 ** 2


def f2(x1, x2):
    return (x1 - 2) ** 2 + x2 ** 2


# 目标函数列表
objectives = [f1, f2]


# ======================
# SPEA 主过程
# ======================
def spea_algorithm():
    # 初始化种群和归档集
    population = adapter_initialize_population(population_size, num_bits, variable_ranges)
    archive = []

    evaluations = 0
    while evaluations < max_evaluations:
        print(f"当前评估次数: {evaluations}")

        # 计算适应度并更新归档集
        for ind in population:
            ind.objectives = adapter_calculate_objectives(ind, objectives, variable_ranges, num_bits)

        archive = update_archive(population, archive, archive_size)
        evaluations += len(population)

        # 生成新的种群
        offspring = create_offspring(archive, variable_ranges, population_size, num_bits)
        population = offspring

    # 返回最终归档集中的解
    final_solutions = [adapter_decode_individual(ind, variable_ranges, num_bits) for ind in archive]
    return final_solutions


# ======================
# 更新归档集
# ======================
def update_archive(population, archive, archive_size):
    combined_population = population + archive
    calculate_strength(combined_population)

    # 基于支配关系更新归档集
    new_archive = [ind for ind in combined_population if ind.fitness < 1]

    if len(new_archive) > archive_size:
        new_archive = crowding_distance_sort(new_archive)[:archive_size]
    elif len(new_archive) < archive_size:
        non_dominated_set = [ind for ind in combined_population if ind not in new_archive]
        new_archive += random.sample(non_dominated_set, archive_size - len(new_archive))

    return new_archive


# ======================
# 适应度计算
# ======================
def calculate_strength(population):
    for ind in population:
        ind.strength = sum(1 for other in population if dominates(ind, other))

    for ind in population:
        ind.fitness = sum(other.strength for other in population if dominates(other, ind))


# ======================
# 支配判断函数
# ======================
def dominates(ind1, ind2):
    better_in_all = True
    strictly_better = False
    for val1, val2 in zip(ind1.objectives, ind2.objectives):
        if val1 > val2:
            better_in_all = False
            break
        elif val1 < val2:
            strictly_better = True
    return better_in_all and strictly_better


# ======================
# 拥挤距离排序
# ======================
def crowding_distance_sort(archive):
    distances = [0] * len(archive)
    num_objectives = len(archive[0].objectives)

    for m in range(num_objectives):
        archive.sort(key=lambda x: x.objectives[m])
        min_obj = archive[0].objectives[m]
        max_obj = archive[-1].objectives[m]
        distances[0] = distances[-1] = float('inf')

        for i in range(1, len(archive) - 1):
            distances[i] += (archive[i + 1].objectives[m] - archive[i - 1].objectives[m])

    for i, ind in enumerate(archive):
        ind.crowding_distance = distances[i]
    archive.sort(key=lambda x: (-x.fitness, -x.crowding_distance))
    return archive


# ======================
# 生成子代
# ======================
def create_offspring(archive, variable_ranges, pop_size, num_bits, crossover_rate=0.9, mutation_rate=0.01):
    offspring = []
    while len(offspring) < pop_size:
        parent1 = tournament_selection(archive, 1)[0]
        parent2 = tournament_selection(archive, 1)[0]

        if random.random() < crossover_rate:
            child1, child2 = adapter_crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2

        offspring.append(adapter_mutate(child1, mutation_rate))
        if len(offspring) < pop_size:
            offspring.append(adapter_mutate(child2, mutation_rate))

    return offspring


# ======================
# 锦标赛选择
# ======================
def tournament_selection(population, size, k=2):
    selected = []
    for _ in range(size):
        tournament = random.sample(population, k)
        sorted_tournament = sorted(tournament, key=lambda ind: (-ind.fitness, ind.crowding_distance))
        selected.append(sorted_tournament[0])
    return selected
def generate_theoretical_pareto_front(num_points=100):
    """
    生成理论帕累托最优解集。

    参数:
    num_points (int): 生成的点的数量，默认为 500。

    返回:
    numpy.ndarray: 理论帕累托最优解集。
    """
    # 对于 f1 和 f2，理论帕累托前沿是 x1 = 1 的线
    # x1 固定为 1，x2 在 -2 到 2 之间均匀分布
    x2_values = np.linspace(-2, 2, num_points)

    # 生成理论帕累托最优解集
    theoretical_front = np.column_stack((np.ones(num_points), x2_values))
    return theoretical_front

# 初始化可视化
visualizer = ObjectiveVisualizer(
    funcs=[f1, f2],
    variable_ranges=variable_ranges,
    show_pareto=True,
    objectives={'f1': 'min', 'f2': 'min'},
    save_gif=save_gif,
    gif_name=gif_name
)

# 执行算法
final_archive = spea_algorithm()
print("最终解集:", final_archive)
optimal_front = generate_theoretical_pareto_front()
caculator1 = DiversityMetricCalculator(final_archive)
caculator2 = ConvergenceMetricCalculator(optimal_front,final_archive,100)
metric1 = caculator1.get_diversity_metric()
metric2 = caculator2.calculate_convergence_metric()
print(metric1)
print(metric2)
# 显示理论帕累托最优解空间和最终解空间
visualizer.draw_populations(final_archive, generation=max_evaluations//population_size)
visualizer.save()  # 保存GIF文件
