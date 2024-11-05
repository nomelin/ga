import numpy as np


class ConvergenceMetricCalculator:
    def __init__(self, optimal_set, obtained_optimal_set, H):
        """
        初始化收敛度量计算器。

        参数:
        optimal_set (numpy.ndarray): 理论最优解集，应为 NumPy 数组。
        obtained_optimal_set (numpy.ndarray): 实际获得的最优解集，应为 NumPy 数组。
        H (int): 用于生成均匀间隔的理论最优解集的点数。
        """
        # 确保输入的解集是 NumPy 数组
        self.optimal_set = np.array(optimal_set)
        self.obtained_optimal_set = np.array(obtained_optimal_set)
        self.H = H

    def generate_uniform_actual_optimum_set(self):
        """
        生成一个均匀间隔的理论最优解集。

        这个解集用于与实际获得的最优解集进行比较，以评估优化算法的性能。
        解集中的每个点的 x1 坐标固定为 1，x2 坐标在 -2 到 2 之间均匀分布。

        返回:
        numpy.ndarray: 均匀间隔的理论最优解集。
        """
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

        参数:
        uniform_actual_optimum_set (numpy.ndarray): 均匀间隔的理论最优解集。

        返回:
        numpy.ndarray: 实际解集中每个点到理论解集的最小距离。
        """
        # 计算实际解集中每个点到理论解集的欧几里得距离
        distances = np.sqrt(
            np.sum((self.obtained_optimal_set[:, np.newaxis, :] - uniform_actual_optimum_set) ** 2, axis=2))
        # 找到每个实际解到理论解集的最小距离
        min_distances = np.min(distances, axis=0)
        return min_distances

    def calculate_convergence_metric(self):
        """
        计算收敛度量，即实际获得的最优解集与理论最优解集之间的平均最小欧几里得距离。

        返回:
        float: 收敛度量，表示实际解集与理论解集之间的平均距离。
        """
        # 生成均匀间隔的理论最优解集
        uniform_actual_optimum_set = self.generate_uniform_actual_optimum_set()
        # 计算实际解集与理论解集之间的最小距离
        min_distances = self.calculate_distance(uniform_actual_optimum_set)
        # 计算最小距离的平均值作为收敛度量
        convergence_metric = np.mean(min_distances)
        return convergence_metric
