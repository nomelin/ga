import matplotlib.pyplot as plt
import numpy as np

"""
ObjectiveVisualizer类用于可视化目标函数的解空间。
"""


class ObjectiveVisualizer:
    def __init__(self, funcs, variable_ranges, resolution, show_pareto=False, objectives=None, figsize=(10, 10)):
        """
        初始化可视化工具，计算目标函数值并生成网格。

        参数:
            funcs (list): 包含两个目标函数的列表，每个函数应接受相同数量的自变量，返回标量目标值。
            variable_ranges (list of tuples): 每个自变量的取值范围列表，格式为 [(x1_min, x1_max), (x2_min, x2_max)]。
            resolution (int): 采样分辨率，表示在每个自变量的范围内采样的点数。
            show_pareto (bool): 是否显示Pareto最优边界，默认为False。
            objectives (dict): 目标优化方向配置字典，例如 {'f1': 'min', 'f2': 'max'}。

        功能:
            初始化时计算目标函数值和Pareto前沿，并保存这些数据以供后续使用。
        """
        self.funcs = funcs
        self.variable_ranges = variable_ranges
        self.resolution = resolution
        self.show_pareto = show_pareto
        self.objectives = objectives
        self.grids = self.generate_grid(variable_ranges, resolution)
        print("[ObjectiveVisualizer]准备计算目标函数值")
        self.F1, self.F2 = self.calculate_objective_values(funcs, self.grids)
        print("[ObjectiveVisualizer]目标函数值计算完成")
        self.pareto_points = None

        if show_pareto and objectives:
            print("[ObjectiveVisualizer]准备计算Pareto前沿")
            self.pareto_points = self.pareto_front(self.F1, self.F2, objectives)
            print("[ObjectiveVisualizer]Pareto前沿点数：", len(self.pareto_points))

        # 初始化figure和axes
        self.fig, self.ax = plt.subplots(figsize=figsize)

    def generate_grid(self, variable_ranges, resolution):
        # 根据分辨率生成网格
        grids = [np.linspace(r[0], r[1], resolution) for r in variable_ranges]
        return np.meshgrid(*grids)

    def calculate_objective_values(self, funcs, grids):
        # 计算所有目标函数的值
        return [func(*grids) for func in funcs]

    def pareto_front(self, F1, F2, objectives):
        # 筛选Pareto前沿
        points = np.vstack((F1.ravel(), F2.ravel())).T
        pareto_points = []

        comp_f1 = (lambda a, b: a <= b) if objectives['f1'] == 'min' else (lambda a, b: a >= b)
        comp_f2 = (lambda a, b: a <= b) if objectives['f2'] == 'min' else (lambda a, b: a >= b)
        strict_comp_f1 = (lambda a, b: a < b) if objectives['f1'] == 'min' else (lambda a, b: a > b)
        strict_comp_f2 = (lambda a, b: a < b) if objectives['f2'] == 'min' else (lambda a, b: a > b)

        for point in points:
            dominated = False
            for other in points:
                if comp_f1(other[0], point[0]) and comp_f2(other[1], point[1]) and \
                        (strict_comp_f1(other[0], point[0]) or strict_comp_f2(other[1], point[1])):
                    dominated = True
                    break
            if not dominated:
                pareto_points.append(point)

        return np.array(pareto_points)

    def draw(self):
        """ 显示图像，不重新计算目标函数值 """
        self.ax.clear()  # 清除当前内容而不重新创建figure
        self.ax.scatter(self.F1, self.F2, s=10, c='lightgray', label='解')
        if self.pareto_points is not None:
            self.ax.scatter(self.pareto_points[:, 0], self.pareto_points[:, 1], c='#a94826', label='最优边界')
        self.ax.set_xlabel('f1')
        self.ax.set_ylabel('f2')
        self.ax.set_title('目标函数解空间')
        self.ax.legend()
        plt.draw()  # 更新当前figure
        print("[ObjectiveVisualizer]重绘目标函数解空间")
