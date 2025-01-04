import numpy as np


class SimilarityDetector:
    def __init__(self, method="objective_difference", threshold=0.1):
        """
        初始化相似性检测算子
        :param method: 检测方法，支持 "objective_difference", "euclidean_distance"
        :param threshold: 判断环境变化的阈值
        """
        self.method = method
        self.threshold = threshold

    def detect(self, current_population, previous_population):
        """
        检测环境是否发生变化（适用于多目标优化）
        :param current_population: 当前种群的目标函数值（N x M 数组，N个个体，M个目标）
        :param previous_population: 上一时刻的目标函数值（N x M 数组）
        :return: True 表示环境发生变化，False 表示没有变化
        """
        if self.method == "objective_difference":
            return self._objective_difference(current_population, previous_population)
        elif self.method == "euclidean_distance":
            return self._euclidean_distance(current_population, previous_population)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _objective_difference(self, current_population, previous_population):
        """
        基于目标函数差异的检测方法（适用于多目标）
        计算每个目标的平均变化程度，取平均
        """
        difference = np.abs(current_population - previous_population)
        mean_difference = np.mean(difference)
        return mean_difference > self.threshold

    def _euclidean_distance(self, current_population, previous_population):
        """
        基于欧氏距离的检测方法（适用于多目标）
        计算每个个体在多目标空间中的欧氏距离
        """
        distances = np.linalg.norm(current_population - previous_population, axis=1)
        mean_distance = np.mean(distances)
        return mean_distance > self.threshold
