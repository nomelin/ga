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
        self.normalized_difference = 0.0  # 用于存储归一化差异值

    def detect(self, current_population, previous_population):
        """
        检测环境是否发生变化，并计算归一化差异
        :param current_population: 当前种群的目标函数值（N x M 数组，N个个体，M个目标）
        :param previous_population: 上一时刻的目标函数值（N x M 数组）
        :return: True 表示环境发生变化，False 表示没有变化
        """
        if self.method == "objective_difference":
            difference = self._objective_difference(current_population, previous_population)
        elif self.method == "euclidean_distance":
            difference = self._euclidean_distance(current_population, previous_population)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        # 归一化差异并计算保留比例
        self.normalized_difference = self._normalize_difference(difference)
        return self.normalized_difference > self.threshold

    def _objective_difference(self, current_population, previous_population):
        """基于目标函数差异的检测方法"""
        difference = np.abs(current_population - previous_population)
        mean_difference = np.mean(difference)
        return mean_difference

    def _euclidean_distance(self, current_population, previous_population):
        """基于欧氏距离的检测方法"""
        distances = np.linalg.norm(current_population - previous_population, axis=1)
        mean_distance = np.mean(distances)
        return mean_distance

    def _normalize_difference(self, difference):
        """对差异进行归一化，限制在 [0, 1] 范围内"""
        return min(1.0, difference / self.threshold)

    def calculate_retention_ratio(self, min_ratio=0.2, max_ratio=1.0):
        """
        根据归一化差异计算保留比例
        :param min_ratio: 最小保留比例
        :param max_ratio: 最大保留比例
        :return: 根据差异计算的保留比例
        """
        # 当归一化差异接近0时，保留比例接近最大；差异增大时，保留比例降低
        retention_ratio = max(min_ratio, max_ratio * (1 - self.normalized_difference))
        return retention_ratio
