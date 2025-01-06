from flask_cors.core import FLASK_CORS_EVALUATED

from lib import global_var
from lib.ga_basic import *
from lib.AdaptabilityMetric import *
from lib.DiversityMetric import *
from lib.SimilarityDetector import *
def nsga2(visualizer, funcs_dict, variable_ranges, precision, pop_size=100, num_generations=50, crossover_rate=0.9,
          mutation_rate=0.01, dynamic_funcs=False,regeneration_ratio=0.2,use_prediction=False):
    """
    NSGA-II 算法主过程，支持动态目标函数。

    参数:
        funcs_dict (dict): 轮次 -> 目标函数列表及其方向。
                           例如 {0: [[f1, f2], ['min', 'min']], 10: [[f3, f4], ['max', 'min']]}。随时间变化的目标函数列表。
        variable_ranges (list of tuples): 每个变量的取值范围。
        precision (float): 期望的搜索精度，用于确定编码长度。
        pop_size (int): 种群大小，默认值为 100。
        num_generations (int): 迭代次数，默认值为 50。
        dynamic_funcs (bool): 是否使用动态目标函数，默认值为 False。
        regeneration_ratio (float): 环境变化时重新生成的种群比例。
        use_prediction (bool): 是否使用预测功能(默认为 False)
        例如：
        def f1(x1, x2, t): return x1 ** 2 + x2 ** 2 + a(t)*x1 + b(t)*x2
        def a(t): return t
        def b(t): return sin(t)
        则这个函数就是参数随时间t变化的目标函数。
        如果启用动态函数，则每个函数都需要在末尾包含一个参数 t，不论是否使用。

    返回:
        list: 最终种群的解（经过解码）。
    """
    global_var.set_algorithm_running(True)  # 设置标志位，表示算法正在运行
    # 初始设定目标函数和方向
    current_funcs, current_directions = funcs_dict[0][0], funcs_dict[0][1]
    # visualizer.reCalculate(current_funcs)

    # 初始化 t 为 0
    if dynamic_funcs:
        t = 0
    else:
        t = None

    # 生成初始种群并进行快速非支配排序
    num_bits = [calculate_num_bits(var_min, var_max, precision) for var_min, var_max in variable_ranges]
    population = adapter_initialize_population(pop_size, num_bits, variable_ranges)

    # 非支配排序
    fronts = fast_non_dominated_sort(population, current_funcs, variable_ranges, num_bits, current_directions,
                                     t)
    # 展平种群
    fronts = [ind for front in fronts for ind in front]
    # 使用选择、交叉和变异生成子代种群
    offspring = create_offspring(fronts, variable_ranges, pop_size, num_bits, crossover_rate, mutation_rate)
    # 初始化相似性检测器（仅在动态环境下启用）
    similarity_detector = SimilarityDetector(threshold=0.1) if dynamic_funcs else None
    previous_objectives = None
    # 初始化适应性度量器
    adaptability_metric = AdaptabilityMetric(threshold=0.1)
    # 初始化动态多样性度量对象
    diversity_metric = DiversityMetric()
    # 用于存储每一代的适应性度量值和多样性变化
    adaptability_scores = []
    diversity_changes = []
    # 迭代进化过程
    for generation in range(num_generations):
        print(f"[nsga-ii] 第 {generation + 1} 代")

        if not global_var.get_algorithm_running():  # 检查标志位
            print("[nsga-ii]NSGA-II 被终止。")
            break

        # 检查是否是分段边界，如果是，则需要更换目标函数
        if generation in funcs_dict:
            if dynamic_funcs:
                t = 0  # 分段时重置 t 为 0
            current_funcs, current_directions = funcs_dict[generation][0], funcs_dict[generation][1]
            print(
                f"[nsga-ii] 分段, 更新目标函数和方向：第 {generation + 1} 代使用新目标 {current_funcs} 和方向 {current_directions}")
            visualizer.reCalculate(funcs=current_funcs, t=t)
            print(f"[nsga-ii] 刷新解空间")
        # 如果使用动态目标函数，每代都重新计算解空间
        if dynamic_funcs:
            print(f"[nsga-ii] 动态函数，刷新解空间。t = {t}")
            visualizer.reCalculate(funcs=current_funcs, t=t)

            # 检查是否发生环境变化
            current_objectives = np.array(
                [adapter_calculate_objectives(ind, current_funcs, variable_ranges, num_bits, t) for ind in population])
            # 打印当前和上一代的目标函数值
            print(f"[nsga-ii] 当前目标函数值: {current_objectives}")
            print(f"[nsga-ii] 上一代目标函数值: {previous_objectives}")

            # 在第一代时，初始化 previous_objectives 为全零目标值
            if previous_objectives is None:
                previous_objectives = np.zeros_like(current_objectives)  # 将目标值初始化为全零数组
                print(f"[nsga-ii] 第一代，初始化 previous_objectives 为全零目标值")
            if previous_objectives is not None and similarity_detector.detect(current_objectives, previous_objectives):
                print(f"[nsga-ii] 检测到环境变化，重新生成种群")
                regeneration_ratio = similarity_detector.calculate_retention_ratio(regeneration_ratio, 1.0)

                # 重新生成一定比例的种群
                if use_prediction:
                    # 使用预测功能（暂时使用随机生成代替）
                    print("[nsga-ii] 使用预测功能生成种群")
                    regenerated_population = adapter_initialize_population(int(pop_size * regeneration_ratio), num_bits,
                                                                           variable_ranges)
                else:
                    # 使用随机生成代替
                    print("[nsga-ii] 使用随机生成代替预测功能")
                    regenerated_population = adapter_initialize_population(int(pop_size * regeneration_ratio), num_bits,
                                                                           variable_ranges)
                # 将重新生成的种群与当前种群合并
                population[:int(pop_size * regeneration_ratio)] = regenerated_population
            # 计算适应性度量
            adaptability_score = adaptability_metric.calculate_adaptation(current_objectives=current_objectives,
                                                                              previous_objectives=previous_objectives,
                                                                              generation=generation)
            adaptability_scores.append(adaptability_score)  # 存储每代的适应性度量值
            print(f"[nsga-ii] 第 {generation + 1} 代的适应性度量值: {adaptability_score:.4f}")
            # 更新当前目标函数值
            previous_objectives = current_objectives

        # 计算多样性变化
        diversity_change = diversity_metric.calculate_population_diversity(offspring, variable_ranges,
                                                                                num_bits)
        diversity_changes.append(diversity_change)  # 存储每代的多样性变化
        print(f"[nsga-ii] 第 {generation + 1} 代的多样性变化: {diversity_change:.4f}")
        # 合并父代和子代生成 2N 个体的种群
        combined_population = population + offspring

        # 非支配排序
        fronts = fast_non_dominated_sort(combined_population, current_funcs, variable_ranges, num_bits,
                                         current_directions, t)
        # 拥挤度排序
        sorted_population = crowding_distance_sort(fronts)

        # 画点
        visualizer.draw_individuals_by_rank(sorted_population, generation)

        # 展平种群
        sorted_population = [ind for front in sorted_population for ind in front]

        # 精英保留策略，从排序后的种群中选择 N 个个体，形成新的父代种群
        population = sorted_population[:pop_size]

        # 使用选择、交叉、变异生成新一代子代种群
        offspring = create_offspring(population, variable_ranges, pop_size, num_bits, crossover_rate, mutation_rate)

        # 更新 t
        if dynamic_funcs:
            t += 1

    # 返回最终种群的解
    final_solutions = [adapter_decode_individual(ind, variable_ranges, num_bits) for ind in population]
    # 在所有代数迭代结束后输出所有适应性度量值和多样性变化
    print("\n nsgaii适应性度量值和多样性变化记录：")
    for i in range(num_generations):
        print(f"第 {i + 1} 代：适应性度量值: {adaptability_scores[i]:.4f}  多样性变化: {diversity_changes[i]:.4f}")

    # 计算并输出平均适应性度量值和平均多样性变化
    average_adaptability_score = sum(adaptability_scores) / len(adaptability_scores) if adaptability_scores else 0
    average_diversity_change = sum(diversity_changes) / len(diversity_changes) if diversity_changes else 0

    print(f"\n平均适应性度量值: {average_adaptability_score:.4f}")
    print(f"平均多样性变化: {average_diversity_change:.4f}")
    return final_solutions


# ======================

def fast_non_dominated_sort(population, funcs, variable_ranges, num_bits, directions, t):
    """
    快速非支配排序，支持每个目标函数的优化方向，并支持动态优化问题中的时间变量 t。

    参数:
        population (list): 当前种群。
        funcs (list): 目标函数列表。
        variable_ranges (list of tuples): 每个变量的取值范围。
        num_bits (list of int): 每个变量的二进制位数。
        directions (list of str): 每个目标的优化方向，'min' 或 'max'。
        t (float): 时间变量，用于动态优化问题。

    返回:
        list[list]: 每个 rank 的解，嵌套 list。
    """
    # print(f"待非支配排序种群: {population}")
    ranks = [[]]
    for i, ind1 in enumerate(population):
        S = []
        n = 0
        for j, ind2 in enumerate(population):
            if dominates(ind1, ind2, funcs, variable_ranges, num_bits, directions, t):
                S.append(j)
            elif dominates(ind2, ind1, funcs, variable_ranges, num_bits, directions, t):
                n += 1
        if n == 0:
            ranks[0].append(i)
            population[i].rank = 0
            population[i].S = S
        else:
            ind1.S = S
            ind1.n = n

    current_rank = 0
    while ranks[current_rank]:
        next_rank = []
        for ind in ranks[current_rank]:
            for j in population[ind].S:
                population[j].n -= 1
                if population[j].n == 0:
                    population[j].rank = current_rank + 1
                    next_rank.append(j)
        ranks.append(next_rank)
        current_rank += 1
    if len(ranks[-1]) == 0:
        ranks.pop()
    print(f"非支配排序后的种群: {ranks}")
    for i in range(len(ranks)):
        for j in range(len(ranks[i])):
            ranks[i][j] = population[ranks[i][j]]
    # print(f"非支配排序后的种群: {ranks}")
    return ranks


def dominates(ind1, ind2, funcs, variable_ranges, num_bits, directions, t):
    """
    判断个体 ind1 是否支配个体 ind2，支持每个目标函数的优化方向，并支持动态优化问题中的时间变量 t。

    参数:
        ind1 (str): 第一个个体的二进制字符串。
        ind2 (str): 第二个个体的二进制字符串。
        funcs (list of functions): 目标函数列表。
        variable_ranges (list of tuples): 每个变量的取值范围。
        num_bits (int): 每个变量的二进制位数。
        directions (list of str): 每个目标的优化方向，'min' 或 'max'。
        t (float): 时间变量，用于动态优化问题。

    返回:
        bool: 如果 ind1 支配 ind2，则返回 True；否则返回 False。
    """
    obj_values1 = adapter_calculate_objectives(ind1, funcs, variable_ranges, num_bits, t)
    obj_values2 = adapter_calculate_objectives(ind2, funcs, variable_ranges, num_bits, t)

    better_in_all_objectives = True  # 在所有目标函数上都有更好的
    strictly_better_in_at_least_one = False  # 在至少一个目标函数上有严格的更好

    for val1, val2, direction in zip(obj_values1, obj_values2, directions):
        if direction == 'min':
            if val1 > val2:
                better_in_all_objectives = False  # 至少有一个目标函数上没有更好
                break
            elif val1 < val2:
                strictly_better_in_at_least_one = True  # 至少有一个目标函数上有严格的更好
        elif direction == 'max':
            if val1 < val2:
                better_in_all_objectives = False
                break
            elif val1 > val2:
                strictly_better_in_at_least_one = True

    return better_in_all_objectives and strictly_better_in_at_least_one


def crowding_distance_sort(fronts):
    """
    拥挤距离排序。

    参数:
        fronts (list): 每个 rank 的解列表。[[]]

    返回:
        list[list]: 拥挤度排序后的解。
    """
    sorted_fronts = []
    for front in fronts:
        distances = [0] * len(front)
        num_objectives = len(front[0].objectives)
        # print(f"当前 front: {front}, 目标数: {num_objectives}")
        for m in range(num_objectives):
            front.sort(key=lambda x: x.objectives[m])  # 按第 m 个目标排序
            min_val = front[0].objectives[m]
            max_val = front[-1].objectives[m]
            distances[0] = distances[-1] = float('inf')  # 边界值
            for i in range(1, len(front) - 1):
                distances[i] += (front[i + 1].objectives[m] - front[i - 1].objectives[m])

        for i, ind in enumerate(front):
            ind.crowding_distance = distances[i]  # 将拥挤度赋值给个体

        sorted_fronts.append(sorted(front, key=lambda x: (-x.rank, -x.crowding_distance)))
    print("拥挤度排序完成")
    # print(f"拥挤度排序后的种群: {sorted_fronts}")
    return sorted_fronts


def create_offspring(population, variable_ranges, pop_size, num_bits, crossover_rate=0.9, mutation_rate=0.01):
    """
    生成子代种群的函数，包括锦标赛选择、交叉和变异操作。

    参数:
        population (list): 当前种群。
        variable_ranges (list of tuples): 每个变量的取值范围。
        pop_size (int): 种群大小。
        num_bits (int): 每个变量的二进制位数。
        crossover_rate (float): 交叉概率。
        mutation_rate (float): 突变概率。

    返回:
        list: 新的子代种群。
    """
    # 锦标赛选择
    offspring = []
    while len(offspring) < pop_size:

        parent1 = tournament_selection(population, 1)[0]
        parent2 = tournament_selection(population, 1)[0]
        if np.random.rand() < crossover_rate:
            child1, child2 = adapter_crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2

        offspring.append(adapter_mutate(child1, mutation_rate))
        if len(offspring) < pop_size:
            offspring.append(adapter_mutate(child2, mutation_rate))

    return offspring


def tournament_selection(population, size, k=2):
    """
    锦标赛选择算子，从种群中选择个体。

    参数:
        population (list): 当前种群。
        pop_size (int): 选择的个体数量。
        k (int): 每次锦标赛中参与的个体数量。

    返回:
        list: 选择后的个体。
    """
    selected = []
    for _ in range(size):
        tournament = random.sample(population, k)
        sorted_tournament = sorted(tournament, key=lambda ind: (
            ind.rank, float('-inf') if ind.crowding_distance is None else -ind.crowding_distance))
        winner = sorted_tournament[0]
        selected.append(winner)
    return selected


class Individual:
    def __init__(self, binary_string, rank=None, crowding_distance=None):
        self.binary_string = binary_string
        self.rank = rank
        self.crowding_distance = crowding_distance
        self.objectives = []
        self.S = []  # 支配该个体的个体的索引列表
        self.n = 0  # 支配该个体的个体的数量

    def __str__(self):
        return f"个体(串={self.binary_string}, rank={self.rank}, 拥挤距离={self.crowding_distance}" \
               f",S={self.S},n={self.n},objectives={self.objectives})\n"

    def __repr__(self):
        return self.__str__()


"""
适配器函数，用于适配 NSGA-II 算法的个体输入格式。
"""


def adapter_binary_encode(value, var_min, var_max, num_bits):
    return binary_encode(value, var_min, var_max, num_bits)


def adapter_binary_decode(individual, var_min, var_max, num_bits):
    return binary_decode(individual.binary_string, var_min, var_max, num_bits)


def adapter_initialize_population(pop_size, num_bits, variable_ranges):
    population_strings = initialize_population(pop_size, num_bits, variable_ranges)
    return [Individual(binary_str) for binary_str in population_strings]


def adapter_decode_individual(individual, variable_ranges, num_bits):
    decoded_values = decode_individual(individual.binary_string, variable_ranges, num_bits)
    return decoded_values


def adapter_calculate_objectives(individual, funcs, variable_ranges, num_bits, t):
    individual.objectives = calculate_objectives(individual.binary_string, funcs, variable_ranges, num_bits, t)
    return individual.objectives


def adapter_crossover(parent1, parent2, crossover_rate=0.9):
    offspring1_str, offspring2_str = crossover(parent1.binary_string, parent2.binary_string, crossover_rate)
    return Individual(offspring1_str), Individual(offspring2_str)


def adapter_mutate(individual, mutation_rate=0.01):
    mutated_str = mutate(individual.binary_string, mutation_rate)
    return Individual(mutated_str)
