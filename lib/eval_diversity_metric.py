import numpy as np


class DynamicDiversityMetricCalculator:
    def __init__(self, distance_threshold=None):
        """
        初始化动态多样性度量计算器。

        参数:
        distance_threshold (float): 用于过滤未收敛点的距离阈值。
        """
        self.distance_threshold = distance_threshold
        self.diversity_metrics = []  # 用于存储每一代的多样性度量值

    def calculate_diversity_metric(self, obtained_optimal_set):
        """
        计算当前代的多样性度量，并存储该值。

        参数:
        obtained_optimal_set (numpy.ndarray): 实际获得的最优解集。

        返回:
        float: 当前代的多样性度量 Δ。
        """
        obtained_optimal_set = np.array(obtained_optimal_set)

        # 过滤未收敛的解
        if self.distance_threshold is not None:
            obtained_optimal_set = self._filter_converged_solutions(obtained_optimal_set)

        # 计算多样性度量
        if len(obtained_optimal_set) < 2:
            diversity_metric = float('inf')
        else:
            diversity_metric = self._compute_diversity(obtained_optimal_set)

        self.diversity_metrics.append(diversity_metric)
        return diversity_metric

    def get_average_diversity(self):
        """
        返回所有迭代代的平均多样性度量。
        """
        if not self.diversity_metrics:
            return float('inf')
        return np.mean(self.diversity_metrics)

    def _filter_converged_solutions(self, obtained_optimal_set):
        """
        过滤掉过于相似的点，去除距离过近的解。
        """
        # 创建距离矩阵
        distances = np.linalg.norm(obtained_optimal_set[:, np.newaxis] - obtained_optimal_set, axis=2)

        # 过滤掉距离小于阈值的解
        mask = np.ones(len(obtained_optimal_set), dtype=bool)
        for i in range(len(obtained_optimal_set)):
            if mask[i]:
                # 计算当前点到所有其他点的距离
                close_points = np.where(distances[i] < self.distance_threshold)[0]
                mask[close_points] = False  # 标记这些点为“相似”点
        return obtained_optimal_set[mask]

    def _compute_diversity(self, obtained_optimal_set):
        """
        计算给定解集的多样性度量。
        """
        # 计算距离时扩展到高维
        distances = np.sqrt(np.sum((obtained_optimal_set[1:] - obtained_optimal_set[:-1]) ** 2, axis=1))
        average_distance = np.mean(distances)

        min_idx = np.argmin(obtained_optimal_set[:, 0])
        max_idx = np.argmax(obtained_optimal_set[:, 0])
        extreme_solutions = obtained_optimal_set[[min_idx, max_idx]]

        df = np.sqrt(np.sum((extreme_solutions[1] - extreme_solutions[0]) ** 2))
        dl = np.sqrt(np.sum((obtained_optimal_set[-1] - obtained_optimal_set[0]) ** 2))

        sum_of_deviations = np.sum(np.abs(distances - average_distance))
        N = len(obtained_optimal_set)
        diversity_metric = (df + dl + sum_of_deviations) / (df + dl + (N - 1) * average_distance)
        return diversity_metric
