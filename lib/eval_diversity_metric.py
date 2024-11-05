import numpy as np


class DiversityMetricCalculator:
    def __init__(self, obtained_optimal_set):
        """
        初始化
        obtained_optimal_set: 得到的最优解集
        """
        # 确保 obtained_optimal_set 是 NumPy 数组
        self.obtained_optimal_set = np.array(obtained_optimal_set)

    # 计算得到的非支配解集中连续解之间的欧几里得距离
    def get_distance_between_consecutive_solutions(self):
        """
        计算得到的非支配解集中连续解之间的欧几里得距离
        """
        # 检查解集是否少于两个，如果是，则返回空数组
        if len(self.obtained_optimal_set) < 2:
            return np.array([])

        distances = np.sqrt(np.sum((self.obtained_optimal_set[1:] - self.obtained_optimal_set[:-1]) ** 2, axis=1))
        return distances

    # 计算平均欧几里得距离
    def get_average_distance(self):
        """
        计算平均欧几里得距离
        """
        distances = self.get_distance_between_consecutive_solutions()
        return np.mean(distances)

    # 计算极值解和多维空间内的边界解
    def get_extreme_and_boundary_sets(self):
        """
        更合理的极值解和边界解计算方法
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

    # 计算多样性度量
    def get_diversity_metric(self):
        """
        计算多样性度量
        """
        average_distance = self.get_average_distance()
        extreme_solutions, boundary_solutions = self.get_extreme_and_boundary_sets()

        # 计算极值解之间的距离
        df = np.sqrt(np.sum((extreme_solutions[1] - extreme_solutions[0]) ** 2))

        # 计算边界解之间的距离
        dl = np.sqrt(np.sum((boundary_solutions[-1] - boundary_solutions[0]) ** 2))  # 取第一个和最后一个作为边界解

        N = len(self.obtained_optimal_set)
        distances = self.get_distance_between_consecutive_solutions()
        sum_of_deviations = np.sum(np.abs(distances - average_distance))

        # 计算多样性度量Δ
        diversity_metric = (df + dl + sum_of_deviations) / (df + dl + (N - 1) * average_distance)

        return diversity_metric

