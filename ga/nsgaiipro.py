from lib import global_var
from lib.SimilarityDetector import SimilarityDetector
import math
from lib.ga_basic import *


def nsga2iipro(visualizer, funcs_dict, variable_ranges, precision, pop_size=100, num_generations=50, crossover_rate=0.9,
               mutation_rate=0.01, dynamic_funcs=False, use_crossover_and_differential_mutation=False, F=0.5,
               regeneration_ratio=0.2, use_prediction=False):
    """
    NSGA-II 算法主过程，支持动态目标函数，并集成环境变化检测和种群预测功能。

    参数:
        funcs_dict (dict): 轮次 -> 目标函数列表及其方向。
                           例如 {0: [[f1, f2], ['min', 'min']], 10: [[f3, f4], ['max', 'min']]}。随时间变化的目标函数列表。
        variable_ranges (list of tuples): 每个变量的取值范围。
        precision (float): 期望的搜索精度，用于确定编码长度。
        pop_size (int): 种群大小，默认值为 100。
        num_generations (int): 迭代次数，默认值为 50。
        dynamic_funcs (bool): 是否使用动态目标函数，默认值为 False。
        use_crossover_and_differential_mutation (bool): 是否使用差分交叉与变异。
        F (float): 差分变异控制参数。
        regeneration_ratio (float): 环境变化时重新生成的种群比例。
        use_prediction (bool): 是否使用预测功能（默认为 False）。
        use_crossover_and_differential_mutation: 是否使用差分交叉函数，默认为False
        F: 差分算子
        例如：
        def f1(x1, x2, t): return x1 ** 2 + x2 ** 2 + a(t)*x1 + b(t)*x2
        def a(t): return t
        def b(t): return sin(t)
        则这个函数就是参数随时间t变化的目标函数。
        如果启用动态函数，则每个函数都需要在末尾包含一个参数 t，不论是否使用。

    返回:
        list: 最终种群的解（经过解码）。
    """
    # 初始设定目标函数和方向
    current_funcs, current_directions = funcs_dict[0][0], funcs_dict[0][1]

    global_var.set_algorithm_running(True)  # 设置标志位，表示算法正在运行

    # 初始化 t 为 0
    if dynamic_funcs:
        t = 0
    else:
        t = None

    # 生成初始种群并进行快速非支配排序
    num_bits = [calculate_num_bits(var_min, var_max, precision) for var_min, var_max in variable_ranges]
    population = adapter_initialize_population(pop_size, num_bits, variable_ranges)

    # 非支配排序
    fronts = fast_non_dominated_sort(population, current_funcs, variable_ranges, num_bits, current_directions, t)
    # 展平种群
    fronts = [ind for front in fronts for ind in front]
    # 使用选择、交叉和变异生成子代种群
    offspring = create_offspring(fronts, variable_ranges, pop_size, num_bits, crossover_rate, mutation_rate)

    # 初始化相似性检测器（仅在动态环境下启用）
    similarity_detector = SimilarityDetector(threshold=0.1) if dynamic_funcs else None
    previous_objectives = None

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

        # 如果使用动态目标函数，每代都重新计算解空间，并进行环境变化检测
        if dynamic_funcs:
            print(f"[nsga-ii] 动态函数，刷新解空间。t = {t}")
            visualizer.reCalculate(funcs=current_funcs, t=t)

            # 检查是否发生环境变化
            current_objectives = np.array(
                [adapter_calculate_objectives(ind, current_funcs, variable_ranges, num_bits, t) for ind in population])
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

            # 更新当前目标函数值
            previous_objectives = current_objectives

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

        # 如果启用差分交叉变异，对子代进行差分交叉变异
        if use_crossover_and_differential_mutation:
            print(f"[nsga-ii] 使用差分变异，参数 F = {F}")
            offspring = crossover_and_differential_mutation(offspring, F=F,crossover_rate=crossover_rate,generation=generation,
                                                            variable_ranges=variable_ranges, precision=precision, pop_size=pop_size,funcs_dict=funcs_dict,t=t)

        # 更新 t
        if dynamic_funcs:
            t += 1

    # 返回最终种群的解
    final_solutions = [adapter_decode_individual(ind, variable_ranges, num_bits) for ind in population]
    return final_solutions


# ======================

# 新增————————
def random_selection(population, num_select):
    """
    随机选择策略：从种群中随机选择指定数量的个体。
    """
    return random.sample(population, num_select)

def crowding_distance_selection(population, num_select, funcs, variable_ranges, num_bits, directions, t):
    """
    基于拥挤度选择策略：优先选择拥挤距离较大的个体，维护种群多样性。
    """
    # 首先，进行非支配排序，将种群分成多个前沿
    fronts = fast_non_dominated_sort(population, funcs, variable_ranges, num_bits, directions, t)  # 获取非支配排序结果

    # 对每个前沿进行拥挤度排序
    sorted_fronts = crowding_distance_sort(fronts)

    selected_population = []
    remaining = num_select

    # 选择个体，首先选择最前沿的个体，直到选满
    for front in sorted_fronts:
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected_population.extend(front)
            remaining -= len(front)
        else:
            # 如果当前前沿个体多于剩余需要的个体，则按拥挤度排序，选择拥挤度最大的个体
            front_sorted_by_distance = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
            selected_population.extend(front_sorted_by_distance[:remaining])
            remaining = 0

    return selected_population



def tournament_selection(population, num_select=3, tournament_size=2):
    """
    锦标赛选择策略：通过锦标赛选择指定数量的个体。
    """
    selected = []
    for _ in range(num_select):
        candidates = random.sample(population, tournament_size)
        winner = min(candidates, key=lambda ind: ind.rank)  # 非支配排序优先
        selected.append(winner)
    return selected

# 计算种群平均距离
def calculate_population_average_distance(population, variable_ranges, precision):
    """
    基于决策变量（解码后的值）计算种群的平均欧几里得距离。

    参数：
        population (list): 当前种群，包含所有个体。
        variable_ranges (list): 每个变量的取值范围。
        precision (float): 编码精度。

    返回：
        float: 种群中所有个体之间的平均距离。
    """
    num_bits = [calculate_num_bits(r[0], r[1], precision) for r in variable_ranges]
    total_distance = 0
    num_individuals = len(population)


    # 计算每对个体之间的欧几里得距离
    for i in range(num_individuals):
        for j in range(i + 1, num_individuals):
            # 获取个体 i 和个体 j 的解码后的实数值
            decoded_i = decode_individual(population[i].binary_string, variable_ranges, num_bits)
            decoded_j = decode_individual(population[j].binary_string, variable_ranges, num_bits)

            # 计算欧几里得距离
            distance = math.sqrt(sum((decoded_i[k] - decoded_j[k]) ** 2 for k in range(len(decoded_i))))
            total_distance += distance

    # 计算平均距离
    num_pairs = num_individuals * (num_individuals - 1) / 2  # 计算所有个体的距离对数
    average_distance = total_distance / num_pairs if num_pairs > 0 else 0

    return average_distance

# 动态选择个体选择策略
def dynamic_selection_strategy_based_on_distance(population, generation, num_generations, variable_ranges, precision):
    """
    基于种群的平均距离来动态选择个体选择策略。

    参数：
        population (list): 当前种群，包含所有个体。
        generation (int): 当前代数。
        num_generations (int): 最大代数。
        variable_ranges (list): 每个变量的取值范围。
        precision (float): 编码精度。

    返回：
        str: 当前选择的个体选择策略。
    """
    # 计算种群的平均距离
    avg_distance = calculate_population_average_distance(population, variable_ranges, precision)
    print(f"Average distance: {avg_distance}")

    # 根据平均距离切换选择策略
    if avg_distance > 1.5:  # 较大值表示多样性较高
        if generation < num_generations * 0.3:
            return 'crowding_distance_selection'  # 初期阶段，平衡收敛与多样性
        else:
            return 'tournament_selection'  # 后期阶段，加速收敛
    else:  # 如果种群多样性较低，增大收敛力度
        return 'tournament_selection'  # 加速收敛


# 差分变异函数
def crossover_and_differential_mutation(
        population,
        F=0.5,
        crossover_rate=0.9,
        generation=None,
        num_generations=50,
        num_select=3,
        tournament_size=2,
        variable_ranges=None,
        precision=None,
        pop_size=None,
        funcs_dict=None,
        t=None,
):
    """
    差分交叉变异操作，先进行差分变异，再根据交叉概率进行交叉操作。

    参数:
        population (list): 当前种群。
        F (float): 差分放缩因子，控制步长。
        crossover_rate (float): 交叉概率，控制交叉操作的发生。
        generation: 当前代数
        num_generation: 最大迭代次数
        num_select (int): 选择的个体数量，默认为3。
        variable_ranges (list): 每个变量的取值范围。
        precision (float): 编码精度。
        pop_size (int): 需要生成的个体数量。

    返回:
        list: 包含多个变异和交叉后的新个体的列表。
    """
    funcs, directions = funcs_dict[0]  # 选择第一个优化问题（可以根据需要调整）
    # 存储生成的变异和交叉个体
    offspring = []
    # 基于种群的平均距离动态选择个体选择策略
    selected_strategy = dynamic_selection_strategy_based_on_distance(population, generation, num_generations,
                                                                     variable_ranges, precision)
    # 动态计算 num_bits
    num_bits = [calculate_num_bits(r[0], r[1], precision) for r in variable_ranges]

    # 遍历 variable_ranges 获取 var_min 和 var_max 列表
    var_min = [r[0] for r in variable_ranges]
    var_max = [r[1] for r in variable_ranges]

    # 生成 pop_size 个变异交叉后的个体
    for _ in range(pop_size):
        # 获取动态选择的策略
        if selected_strategy == 'random_selection':
            selected = random_selection(population, num_select)  # 只需要 population 和 num_select
        elif selected_strategy == 'crowding_distance_selection':
            selected = crowding_distance_selection(population, num_select, funcs, variable_ranges, num_bits, directions,
                                                   t)  # 需要 funcs, variable_ranges, num_bits, directions, t
        elif selected_strategy == 'tournament_selection':
            selected = tournament_selection(population, num_select, tournament_size)  # 需要 tournament_size 参数（默认为 2）

        # 将选中的个体分配给 a, b, c
        a, b, c = selected

        # 解码 a, b, c 的二进制字符串为实数值
        decoded_a = decode_individual(a.binary_string, variable_ranges, num_bits)
        decoded_b = decode_individual(b.binary_string, variable_ranges, num_bits)
        decoded_c = decode_individual(c.binary_string, variable_ranges, num_bits)

        # 进行差分变异运算，生成一个变异体
        donor = [decoded_a[i] + F * (decoded_b[i] - decoded_c[i]) for i in range(len(decoded_a))]

        # 限制 donor 的值在 variable_ranges 范围内
        donor = [min(max(donor[i], var_min[i]), var_max[i]) for i in range(len(donor))]

        # 使用 encode_individual 编码 donor
        encoded_donor = encode_individual(donor, var_min=min(var_min), var_max=max(var_max), precision=precision)

        # 进行交叉操作，结合 donor 和父代个体生成新的个体
        if random.random() < crossover_rate:
            # 执行交叉操作
            offspring1, offspring2 = crossover(encoded_donor, a.binary_string, crossover_rate)
        else:
            # 如果不交叉，直接将变异体加入 offspring
            offspring1 = encoded_donor
            offspring2 = encoded_donor

        # 将交叉后的个体加入 offspring 列表
        offspring.append(Individual(binary_string=offspring1))
        offspring.append(Individual(binary_string=offspring2))
        print(len(offspring))

    return offspring


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
        population (list[list]): 当前种群。
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
