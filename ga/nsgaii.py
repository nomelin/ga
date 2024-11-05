from lib.ga_basic import *


# ======================
# 参数配置
# ======================
# 函数
def f1(x1, x2):
    return x1 ** 2 + x2 ** 2


def f2(x1, x2):
    return (x1 - 2) ** 2 + x2 ** 2


# 自变量范围和取值分辨率
variable_ranges = [(-2, 2), (-2, 2)]
resolution = 500  # 离散的取值数量


# =======================
# 专用函数
# =======================

def non_dominated_sort(population, funcs, variable_ranges, num_bits):
    """执行非支配排序并计算拥挤距离。"""
    obj_values = [calculate_objectives(ind, funcs, variable_ranges, num_bits) for ind in population]
    # 排序和计算拥挤距离的具体逻辑实现
    # 返回按前沿层分组的种群以及每个个体的拥挤距离
    pass  # 实现略，需计算非支配层次和拥挤距离


def tournament_selection(population, num_winners):
    """锦标赛选择操作。"""
    winners = []
    for _ in range(num_winners):
        a, b = random.sample(population, 2)
        # 根据非支配层和拥挤距离选择优胜者
        # 假设包含非支配层次和拥挤距离信息
        pass  # 实现略
    return winners


# ======================
# NSGA-II 主过程
# ======================
def nsga2(funcs, variable_ranges, precision, pop_size=100, num_generations=50):
    """
    NSGA-II 算法主过程。

    参数:
        funcs (list): 目标函数列表。
        variable_ranges (list of tuples): 每个变量的取值范围。
        precision (float): 期望的搜索精度，用于确定编码长度。
        pop_size (int): 种群大小，默认值为 100。
        num_generations (int): 迭代次数，默认值为 50。

    返回:
        list: 最终种群的解（经过解码）。
    """
    # 计算每个变量的二进制编码位数
    num_bits = [calculate_num_bits(var_min, var_max, precision) for var_min, var_max in variable_ranges]

    # Step 1: 生成初始种群并进行快速非支配排序
    population = initialize_population(pop_size, num_bits, variable_ranges)
    fronts = non_dominated_sort(population, funcs, variable_ranges, num_bits)
    population_with_crowding = crowding_distance_selection(fronts, pop_size)

    # Step 2: 使用锦标赛选择生成初始子代种群
    offspring = []
    while len(offspring) < pop_size:
        parent1, parent2 = tournament_selection(population_with_crowding, pop_size)
        child1, child2 = crossover(parent1, parent2)
        offspring.append(mutate(child1))
        if len(offspring) < pop_size:
            offspring.append(mutate(child2))

    # Step 3: 迭代进化过程
    for generation in range(num_generations):
        print(f"第 {generation + 1} 代")

        # 将父代和子代合并生成 2N 个体的种群
        combined_population = population + offspring

        # 非支配排序并计算拥挤度
        fronts = non_dominated_sort(combined_population, funcs, variable_ranges, num_bits)
        population_with_crowding = crowding_distance_selection(fronts, pop_size)

        # 精英策略选择出新的父代种群
        population = population_with_crowding[:pop_size]

        # Step 4: 使用选择、交叉、变异生成新一代子代种群
        offspring = []
        while len(offspring) < pop_size:
            parent1, parent2 = tournament_selection(population, pop_size)
            child1, child2 = crossover(parent1, parent2)
            offspring.append(mutate(child1))
            if len(offspring) < pop_size:
                offspring.append(mutate(child2))

    # 返回最终种群的解
    final_solutions = [decode_individual(ind, variable_ranges, num_bits) for ind in population]
    return final_solutions
