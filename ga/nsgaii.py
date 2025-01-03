from lib.ga_basic import *
from sklearn.neural_network import MLPRegressor

# ======================
# NSGA-II 主过程
# ======================
def nsga2(visualizer, funcs_dict, variable_ranges, precision, pop_size=100, num_generations=50, crossover_rate=0.9,
          mutation_rate=0.01):
    """
    NSGA-II 算法主过程，支持动态目标函数。

    参数:
        funcs_dict (dict): 轮次 -> 目标函数列表及其方向。
                           例如 {0: [[f1, f2], ['min', 'min']], 10: [[f3, f4], ['max', 'min']]}。随时间变化的目标函数列表。
        variable_ranges (list of tuples): 每个变量的取值范围。
        precision (float): 期望的搜索精度，用于确定编码长度。
        pop_size (int): 种群大小，默认值为 100。
        num_generations (int): 迭代次数，默认值为 50。

    返回:
        list: 最终种群的解（经过解码）。
    """
    # 初始设定目标函数和方向
    current_funcs, current_directions = funcs_dict[0][0], funcs_dict[0][1]

    # 生成初始种群并进行快速非支配排序
    num_bits = [calculate_num_bits(var_min, var_max, precision) for var_min, var_max in variable_ranges]
    population = adapter_initialize_population(pop_size, num_bits, variable_ranges)

    # 非支配排序
    fronts = fast_non_dominated_sort(population, current_funcs, variable_ranges, num_bits, current_directions)
    # 展平种群
    fronts = [ind for front in fronts for ind in front]
    # 初始化保存历史解的列表
    historical_fronts = []  # 新增——————

    # 使用选择、交叉和变异生成子代种群
    # offspring = create_offspring(fronts, variable_ranges, pop_size, num_bits, crossover_rate, mutation_rate)

    # 使用预测生成初始种群和遗传操作生成新一代子代种群
    offspring = create_offspring_with_prediction(
        population, variable_ranges, pop_size, num_bits, historical_fronts,
        crossover_rate, mutation_rate, de_strategy='rank', F=0.5
    )
    # 新增——————

    # 初始化保存历史解的列表
    historical_fronts = []  # 新增——————
    # 迭代进化过程
    for generation in range(num_generations):
        print(f"[nsga-ii] 第 {generation + 1} 代")

        # 检查是否需要更换目标函数
        if generation in funcs_dict:
            current_funcs, current_directions = funcs_dict[generation][0], funcs_dict[generation][1]
            print(
                f"[nsga-ii]更新目标函数和方向：第 {generation + 1} 代使用新目标 {current_funcs} 和方向 {current_directions}")

        # 合并父代和子代生成 2N 个体的种群
        combined_population = population + offspring

        # 非支配排序
        fronts = fast_non_dominated_sort(combined_population, current_funcs, variable_ranges, num_bits,
                                         current_directions)
        # 保存当前非支配解集
        historical_fronts.append([adapter_decode_individual(ind, variable_ranges, num_bits) for ind in fronts[0]])
        #  新增——————

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

    # 返回最终种群的解
    final_solutions = [adapter_decode_individual(ind, variable_ranges, num_bits) for ind in population]
    return final_solutions


# ======================

# 差分交叉函数
# 新增————————
def differential_crossover(population, variable_ranges, F=0.5, strategy='rank'):
    """
    差分交叉算子，用于生成变异个体。

    参数:
        population (list): 当前种群。
        variable_ranges (list of tuples): 每个变量的取值范围。
        F (float): 差分缩放因子，通常取值在 [0.4, 0.9]。
        strategy (str): 个体选择策略，可选值为 'random', 'crowding', 'rank', 'tournament'。

    返回:
        Individual: 生成的变异个体。
    """
    pop_size = len(population)
    if strategy == 'random':
        # 随机选择 3 个不同的个体
        r1, r2, r3 = np.random.choice(range(pop_size), 3, replace=False)
    elif strategy == 'crowding':
        # 根据拥挤距离选择，优先选择拥挤度较低的个体
        sorted_pop = sorted(population, key=lambda ind: ind.crowding_distance, reverse=True)
        r1, r2, r3 = np.random.choice(range(len(sorted_pop)), 3, replace=False)
    elif strategy == 'rank':
        # 根据非支配排序优先选择 rank 较低的个体
        sorted_pop = sorted(population, key=lambda ind: ind.rank)
        r1, r2, r3 = np.random.choice(range(len(sorted_pop)), 3, replace=False)
    elif strategy == 'tournament':
        # 锦标赛选择
        r1 = tournament_selection(population, 1)[0]
        r2 = tournament_selection(population, 1)[0]
        r3 = tournament_selection(population, 1)[0]
    else:
        raise ValueError(f"未知的个体选择策略: {strategy}")

    # 计算变异向量
    mutant = [
        population[r1].binary_string[i] + F * (population[r2].binary_string[i] - population[r3].binary_string[i])
        for i in range(len(population[0].binary_string))
    ]

    # 约束修正：确保变异个体在变量范围内
    for i, (var_min, var_max) in enumerate(variable_ranges):
        mutant[i] = np.clip(mutant[i], var_min, var_max)

    return Individual(mutant)


# 使用历史非支配解集，通过预测模型生成部分子代。用遗传操作补充生成剩余的子代。
# 新增——————————
def create_offspring_with_prediction(population, variable_ranges, pop_size, num_bits, historical_fronts,
                                      crossover_rate=0.9, mutation_rate=0.01, de_strategy='rank', F=0.5):
    """
    使用差分交叉算子结合预测模型生成子代种群。

    参数:
        population (list): 当前种群。
        variable_ranges (list of tuples): 每个变量的取值范围。
        pop_size (int): 子代种群大小。
        num_bits (list of int): 每个变量的二进制位数。
        historical_fronts (list): 保存的历史非支配解集。
        crossover_rate (float): 交叉概率。
        mutation_rate (float): 突变概率。
        de_strategy (str): 差分交叉的个体选择策略。
        F (float): 差分交叉的缩放因子。

    返回:
        list: 新的子代种群。
    """
    # 使用历史解生成预测解
    if len(historical_fronts) >= 3:
        recent_fronts = historical_fronts[-3:]
        predicted_solutions = predict_new_solutions(recent_fronts, pop_size // 2, variable_ranges)
    else:
        predicted_solutions = []

    # 转换预测解为个体
    predicted_individuals = [
        Individual(adapter_binary_encode(sol, var_min, var_max, num_bits))
        for sol, (var_min, var_max, num_bits) in zip(predicted_solutions, variable_ranges, num_bits)
    ]

    # 用差分交叉生成其余子代
    offspring = []
    while len(offspring) < pop_size - len(predicted_individuals):
        child = differential_crossover(population, variable_ranges, F=F, strategy=de_strategy)
        offspring.append(child)

    return predicted_individuals + offspring


# 使用历史非支配解集训练预测模型，生成下一代的解
# 新增————————
def predict_new_solutions(recent_fronts, num_solutions, variable_ranges):
    """
    使用历史非支配解集预测新种群解。

    参数:
        recent_fronts (list): 最近几代的非支配解集。
        num_solutions (int): 需要预测的解数量。
        variable_ranges (list of tuples): 每个变量的取值范围。

    返回:
        list: 预测的解。
    """
    # 数据准备：将最近几代的解平铺成训练集
    X_train = []
    Y_train = []
    for i in range(len(recent_fronts) - 1):
        X_train.extend(recent_fronts[i])  # 当前代解
        Y_train.extend(recent_fronts[i + 1])  # 下一代解

    # 训练回归模型
    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
    model.fit(X_train, Y_train)

    # 使用模型预测新解
    predicted_solutions = model.predict(recent_fronts[-1])  # 最近一代预测下一代
    predicted_solutions = np.clip(predicted_solutions, [r[0] for r in variable_ranges],
                                   [r[1] for r in variable_ranges])  # 限制在范围内

    return predicted_solutions[:num_solutions]



def fast_non_dominated_sort(population, funcs, variable_ranges, num_bits, directions):
    """
    快速非支配排序，支持每个目标函数的优化方向。

    参数:
        population (list): 当前种群。
        funcs (list): 目标函数列表。
        variable_ranges (list of tuples): 每个变量的取值范围。
        num_bits (list of int): 每个变量的二进制位数。
        directions (list of str): 每个目标的优化方向，'min' 或 'max'。

    返回:
        list[list]: 每个 rank 的解，嵌套 list。
    """
    # print(f"待非支配排序种群: {population}")
    ranks = [[]]
    for i, ind1 in enumerate(population):
        S = []
        n = 0
        for j, ind2 in enumerate(population):
            if dominates(ind1, ind2, funcs, variable_ranges, num_bits, directions):
                S.append(j)
            elif dominates(ind2, ind1, funcs, variable_ranges, num_bits, directions):
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


def dominates(ind1, ind2, funcs, variable_ranges, num_bits, directions):
    """
    判断个体 ind1 是否支配个体 ind2，支持每个目标函数的优化方向。

    参数:
        ind1 (str): 第一个个体的二进制字符串。
        ind2 (str): 第二个个体的二进制字符串。
        funcs (list of functions): 目标函数列表。
        variable_ranges (list of tuples): 每个变量的取值范围。
        num_bits (int): 每个变量的二进制位数。
        directions (list of str): 每个目标的优化方向，'min' 或 'max'。

    返回:
        bool: 如果 ind1 支配 ind2，则返回 True；否则返回 False。
    """
    obj_values1 = adapter_calculate_objectives(ind1, funcs, variable_ranges, num_bits)
    obj_values2 = adapter_calculate_objectives(ind2, funcs, variable_ranges, num_bits)

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


def adapter_calculate_objectives(individual, funcs, variable_ranges, num_bits):
    individual.objectives = calculate_objectives(individual.binary_string, funcs, variable_ranges, num_bits)
    return individual.objectives


def adapter_crossover(parent1, parent2, crossover_rate=0.9):
    offspring1_str, offspring2_str = crossover(parent1.binary_string, parent2.binary_string, crossover_rate)
    return Individual(offspring1_str), Individual(offspring2_str)


def adapter_mutate(individual, mutation_rate=0.01):
    mutated_str = mutate(individual.binary_string, mutation_rate)
    return Individual(mutated_str)
