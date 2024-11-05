import numpy as np


class ConvergenceMetricCalculator:
    def __init__(self, optimum_front, obtained_optimum_set, H):
        """
        初始化收敛度量计算器
        optimum_front: 最优解集，表示实际的理想解
        obtained_optimum_set: 得到的最优解集
        H: 均匀间隔解集的个数
        """
        self.optimum_front = np.array(optimum_front)
        self.obtained_optimum_set = np.array(obtained_optimum_set)
        self.H = H

    def generate_uniform_actual_optimum_set(self):
        """
        生成均匀间隔的实际最优解集
        """
        uniform_actual_optimum_set = np.linspace(self.optimum_front.min(), self.optimum_front.max(), self.H)
        return uniform_actual_optimum_set

    def calculate_distance(self, uniform_actual_optimum_set):
        """
        计算得到的最优解集与实际均匀间隔的最优解集的最小欧几里得距离
        """
        min_distances = np.min(
            np.sqrt(np.sum((self.obtained_optimum_set - uniform_actual_optimum_set[:, np.newaxis]) ** 2, axis=2)),
            axis=0
        )
        return min_distances

    def calculate_convergence_metric(self):
        """
        计算收敛度量（最小欧几里得距离的平均值）
        """
        uniform_actual_optimum_set = self.generate_uniform_actual_optimum_set()
        min_distances = self.calculate_distance(uniform_actual_optimum_set)
        convergence_metric = np.mean(min_distances)
        return convergence_metric


# 示例用法：
if __name__ == "__main__":
    optimum_front = np.array([[0, 1], [1, 2], [2, 3]])  # 示例最优解集
    obtained_optimum_set = np.random.rand(10, 2)  # 随机生成的示例得到的最优解集
    H = 5  # 均匀间隔的解集个数

    # 创建收敛度量计算器对象
    calculator = ConvergenceMetricCalculator(optimum_front, obtained_optimum_set, H)

    # 计算收敛度量
    convergence_metric = calculator.calculate_convergence_metric()
    print(f"Convergence Metric: {convergence_metric}")
