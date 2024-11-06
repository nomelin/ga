import numpy as np


class DiversityMetricCalculator:
    def __init__(self, obtained_optimal_set, distance_threshold=None):
        """
        初始化多样性度量计算器。

        参数:
        obtained_optimal_set (numpy.ndarray): 实际获得的最优解集，应为 NumPy 数组。
        distance_threshold (float): 距离阈值，用于过滤未收敛的点。
        """
        # 确保 obtained_optimal_set 是 NumPy 数组
        self.obtained_optimal_set = np.array(obtained_optimal_set)
        self.distance_threshold = distance_threshold

    def filter_converged_solutions(self, optimal_set):
        """
        过滤掉未收敛的点，只保留与理论解集接近的解。

        参数:
        optimal_set (numpy.ndarray): 理论最优解集，用于过滤未收敛点。

        返回:
        numpy.ndarray: 过滤后的收敛解集。
        """
        if self.distance_threshold is None:
            return self.obtained_optimal_set

        # 使用 cKDTree 计算每个实际解点到理论解集的最小距离
        from scipy.spatial import cKDTree
        tree = cKDTree(optimal_set)
        min_distances, _ = tree.query(self.obtained_optimal_set)

        # 过滤掉距离大于阈值的解
        return self.obtained_optimal_set[min_distances <= self.distance_threshold]

    def get_distance_between_consecutive_solutions(self):
        """
        计算得到的非支配解集中连续解之间的欧几里得距离。
        """
        # 检查解集是否少于两个，如果是，则返回空数组
        if len(self.obtained_optimal_set) < 2:
            return np.array([])

        distances = np.sqrt(np.sum((self.obtained_optimal_set[1:] - self.obtained_optimal_set[:-1]) ** 2, axis=1))
        return distances

    def get_average_distance(self):
        """
        计算平均欧几里得距离。
        """
        distances = self.get_distance_between_consecutive_solutions()
        return np.mean(distances)

    def get_extreme_and_boundary_sets(self):
        """
        更合理的极值解和边界解计算方法。
        """
        # 找到第一个目标维度上的极值解
        min_idx = np.argmin(self.obtained_optimal_set[:, 0])
        max_idx = np.argmax(self.obtained_optimal_set[:, 0])
        extreme_solutions = self.obtained_optimal_set[[min_idx, max_idx]]

        # 遍历每个维度并找到每个维度的边界解
        boundary_solutions = []
        num_dimensions = self.obtained_optimal_set.shape[1]

        for dim in range(num_dimensions):
            min_idx = np.argmin(self.obtained_optimal_set[:, dim])
            max_idx = np.argmax(self.obtained_optimal_set[:, dim])
            boundary_solutions.append(self.obtained_optimal_set[min_idx])
            boundary_solutions.append(self.obtained_optimal_set[max_idx])

        # 删除重复的边界解
        boundary_solutions = np.unique(boundary_solutions, axis=0)

        return extreme_solutions, boundary_solutions

    def get_diversity_metric(self, optimal_set=None):
        """
        计算多样性度量。

        参数:
        optimal_set (numpy.ndarray): 理论最优解集（如果提供），用于过滤未收敛点。

        返回:
        float: 多样性度量 Δ。
        """
        # 如果提供了理论最优解集，则过滤未收敛的解
        if optimal_set is not None:
            self.obtained_optimal_set = self.filter_converged_solutions(optimal_set)

        # 如果过滤后解集为空，则无法计算多样性度量，返回无穷大
        if len(self.obtained_optimal_set) < 2:
            return float('inf')

        average_distance = self.get_average_distance()
        extreme_solutions, boundary_solutions = self.get_extreme_and_boundary_sets()

        # 计算极值解之间的距离
        df = np.sqrt(np.sum((extreme_solutions[1] - extreme_solutions[0]) ** 2))

        # 计算边界解之间的距离
        dl = np.sqrt(np.sum((boundary_solutions[-1] - boundary_solutions[0]) ** 2))  # 取第一个和最后一个作为边界解

        N = len(self.obtained_optimal_set)
        distances = self.get_distance_between_consecutive_solutions()
        sum_of_deviations = np.sum(np.abs(distances - average_distance))

        # 计算多样性度量 Δ
        diversity_metric = (df + dl + sum_of_deviations) / (df + dl + (N - 1) * average_distance)

        return diversity_metric
