import numpy as np

class AdaptabilityMetric:
    def __init__(self, threshold=0.1):
        """
        初始化适应性度量计算器
        参数:
            threshold: 判断适应性时，使用的阈值（例如：种群与理论前沿的差异阈值）
        """
        self.threshold = threshold
        self.reaction_times = []  # 存储每次反应的时间
        self.previous_objectives = None  # 上一代种群的目标函数值
        self.current_objectives = None  # 当前代种群的目标函数值
        self.reaction_start_gen = None  # 反应开始的代数
        self.adaptation_scores = []  # 存储每代的适应性分数

    def calculate_adaptation(self, current_objectives, previous_objectives, generation):
        """
        计算适应性度量
        参数:
            current_objectives: 当前代种群的目标函数值
            previous_objectives: 上一代种群的目标函数值
            generation: 当前代数
            return: 返回当前代的适应性度量值
        """
        # 如果检测到环境变化，记录反应时间
        if self.previous_objectives is not None and self._detect_change(previous_objectives, current_objectives):
            # 如果之前没有反应开始时间，则记录当前代数作为反应开始代数
            if self.reaction_start_gen is None:
                self.reaction_start_gen = generation

        # 计算适应性度量，基于种群的目标值变化
        adaptation_score = self._calculate_adaptation_score(current_objectives, previous_objectives)
        self.adaptation_scores.append(adaptation_score)

        # 计算反应时间
        if self.reaction_start_gen is not None:
            reaction_time = generation - self.reaction_start_gen
            self.reaction_times.append(reaction_time)

        # 更新上一代种群
        self.previous_objectives = current_objectives

        return adaptation_score

    def _detect_change(self, previous_objectives, current_objectives):
        """
        判断是否发生环境变化，使用MPS度量
        参数:
            previous_objectives: 上一代种群的目标函数值
            current_objectives: 当前代种群的目标函数值
            return: True表示环境发生变化，False表示没有变化
        """
        # 使用MPS(均方根误差)来计算环境变化
        mps = np.mean(np.linalg.norm(current_objectives - previous_objectives, axis=1))
        print(f"[nsga-ii] 当前MPS: {mps}")
        # 如果MPS大于阈值，则判定为环境发生变化
        return mps > self.threshold

    def _calculate_adaptation_score(self, current_objectives, previous_objectives):
        """
        计算适应性分数，可以基于种群目标函数的差异等方式
        参数:
            current_objectives: 当前代种群的目标函数值
            previous_objectives: 上一代种群的目标函数值
            return: 适应性分数
        """
        # 计算当前种群与上一代种群目标函数值的差异，作为适应性分数
        difference = np.mean(np.abs(current_objectives - previous_objectives))  # 按元素计算差异
        return difference