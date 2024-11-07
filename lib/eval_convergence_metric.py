import numpy as np
from scipy.spatial import cKDTree

class ConvergenceMetricCalculator:
    def __init__(self, obtained_optimal_set, H, distance_threshold=None, num_points=20):
        """
        初始化收敛度量计算器。

        参数:
        obtained_optimal_set (numpy.ndarray): 实际获得的最优解集，应为 NumPy 数组。
        H (int): 用于生成均匀间隔的理论最优解集的点数。
        distance_threshold (float): 距离阈值，用于过滤未收敛的点。
        """
        self.num_points = num_points
        self.convergence_metrics = []  # 用于存储每一代的多样性度量值

        self.obtained_optimal_set = np.array(obtained_optimal_set)
        self.H = H
        self.distance_threshold = distance_threshold
        self.optimal_set = None  # 初始化为空，稍后更新

    def update_objective_function(self, objective_func, variable_range=(-2, 2)):
        """
        更新目标函数和自变量范围，并重新生成理论帕累托前沿。

        参数:
        objective_func (function): 新的目标函数。
        variable_range (tuple): 新的自变量范围。
        """
        self.generate_theoretical_pareto_front(objective_func, variable_range)

    def is_dominated(self, point1, point2):
        """
        判断解point1是否支配解point2。
        如果point1在所有目标函数上都小于或等于point2，并且在至少一个目标函数上严格小于point2，则point1支配point2。
        """
        # 对每个目标函数的值进行逐一比较
        return np.all(np.array(point1) <= np.array(point2)) and np.any(np.array(point1) < np.array(point2))

    def generate_theoretical_pareto_front(self, objective_funcs, variable_range=(-2, 2), num_variables=2):
        """
        生成新的理论帕累托前沿，展示多个目标函数的权衡。

        参数:
        objective_funcs (list): 目标函数列表，用于生成帕累托前沿。
        variable_range (tuple): 自变量范围，用于生成帕累托前沿，默认为(-2, 2)。
        num_variables (int): 自变量的数量，默认为2。
        """
        # 为每个自变量定义范围，生成对应的变量值
        variables = [np.linspace(variable_range[0], variable_range[1], self.num_points) for _ in range(num_variables)]

        pareto_front = []

        # 使用itertools.product生成自变量范围的笛卡尔积，表示所有自变量的组合
        from itertools import product
        all_combinations = product(*variables)

        # 遍历所有自变量组合
        for combination in all_combinations:

            # 对每个目标函数进行评估，传入当前的自变量组合
            objective_values = [objective_func(*combination) for objective_func in objective_funcs]

            # 将多个目标函数的值合并为一个点
            current_point = np.array(objective_values)

            # 检查当前点是否被已有的点支配
            dominated = False
            for existing_point in pareto_front:
                if self.is_dominated(existing_point, current_point):  # 如果当前点被支配
                    dominated = True
                    break
                elif self.is_dominated(current_point, existing_point):  # 如果已有点被当前点支配
                    pareto_front.remove(existing_point)

            # 如果当前点没有被支配，则将其加入帕累托前沿
            if not dominated:
                pareto_front.append(current_point)

        # 将帕累托前沿列表转换为numpy数组
        self.optimal_set = np.array(pareto_front)

    def generate_uniform_actual_optimum_set(self):
        """
        生成一个均匀间隔的理论最优解集。

        返回:
        numpy.ndarray: 均匀间隔的理论最优解集。
        """
        if self.optimal_set is None:
            raise ValueError("理论最优解集尚未生成，请调用 update_optimal_set 方法初始化。")

        # 针对每个维度生成均匀分布的点
        uniform_sets = [np.linspace(self.optimal_set[:, dim].min(),
                                    self.optimal_set[:, dim].max(),
                                    self.H) for dim in range(self.optimal_set.shape[1])]

        # 将生成的均匀分布点组合成 (H, n) 形状
        uniform_actual_optimum_set = np.array(np.meshgrid(*uniform_sets)).T.reshape(-1, self.optimal_set.shape[1])
        return uniform_actual_optimum_set

    def calculate_distance(self, uniform_actual_optimum_set):
        """
        计算实际获得的最优解集与均匀间隔的理论最优解集之间的最小欧几里得距离。

        返回:
        numpy.ndarray: 实际解集中每个点到理论解集的最小距离。
        """
        tree = cKDTree(uniform_actual_optimum_set)
        min_distances, _ = tree.query(self.obtained_optimal_set)
        return min_distances

    def calculate_convergence_metric(self):
        """
        计算收敛度量，即实际获得的最优解集与理论最优解集之间的平均最小欧几里得距离。
        会过滤掉距离较大的点（未收敛点）。

        返回:
        float: 收敛度量，表示实际解集与理论解集之间的平均距离。
        """
        # 生成均匀间隔的理论最优解集
        uniform_actual_optimum_set = self.generate_uniform_actual_optimum_set()
        # 计算实际解集与理论解集之间的最小距离
        min_distances = self.calculate_distance(uniform_actual_optimum_set)

        # 如果设置了距离阈值，则过滤掉距离超过阈值的点
        if self.distance_threshold is not None:
            min_distances = min_distances[min_distances <= self.distance_threshold]

        # 计算最小距离的平均值作为收敛度量
        convergence_metric = np.mean(min_distances) if len(min_distances) > 0 else float('inf')
        self.convergence_metrics.append(convergence_metric)
        return convergence_metric
