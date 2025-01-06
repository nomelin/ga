import numpy as np
from lib.ga_basic import *

# 动态多样性度量类
class DiversityMetric:
    def __init__(self):
        self.previous_diversity = 0  # 上一代的多样性
        self.current_diversity = 0  # 当前代的多样性

    def euclidean_distance(self, individual1, individual2):
        """
        计算两个个体之间的欧几里得距离
        """
        return np.linalg.norm(np.array(individual1) - np.array(individual2))

    def calculate_population_diversity(self, population, variable_ranges, num_bits):
        """
        计算种群的多样性：通过计算种群中所有个体之间的欧几里得距离的平均值
        """
        diversity = 0.0
        count = 0
        for i in range(len(population)):
            # 解码个体
            individual1 = decode_individual(population[i].binary_string, variable_ranges, num_bits)
            for j in range(i + 1, len(population)):
                # 解码第二个个体
                individual2 = decode_individual(population[j].binary_string, variable_ranges, num_bits)
                diversity += self.euclidean_distance(individual1, individual2)
                count += 1
        return diversity / count if count > 0 else 0



