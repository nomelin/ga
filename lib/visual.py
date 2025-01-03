import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from lib.tool import GIFGenerator

"""
ObjectiveVisualizer类用于可视化目标函数的解空间。
"""


class ObjectiveVisualizer:
    def __init__(self, funcs, variable_ranges, resolution=100, show_pareto=False, objectives=None, figsize=(8, 8),
                 visual_mode=1, save_gif=False, gif_name='default'):
        """
        初始化可视化工具，计算目标函数值并生成网格。

        参数:
            funcs (list): 包含两个目标函数的列表，每个函数应接受相同数量的自变量，返回标量目标值。
            variable_ranges (list of tuples): 每个自变量的取值范围列表，格式为 [(x1_min, x1_max), (x2_min, x2_max)]。
            resolution (int): 采样分辨率，表示在每个自变量的范围内采样的点数。
            show_pareto (bool): 是否显示Pareto最优边界，默认为False。
            objectives (dict): 目标优化方向配置字典，例如 {'f1': 'min', 'f2': 'max'}。
            visual_mode 1:使用matplotlib;

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
        if save_gif:
            self.gif_generator = GIFGenerator(gif_name + ".gif")
        self.visual_mode = visual_mode
        self.save_gif = save_gif

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
        for i, point in enumerate(points):
            print(f"\r正在计算第{i + 1}/{len(points)}个点是否为Pareto最优点", end='')
            dominated = False
            for other in points:
                if comp_f1(other[0], point[0]) and comp_f2(other[1], point[1]) and \
                        (strict_comp_f1(other[0], point[0]) or strict_comp_f2(other[1], point[1])):
                    dominated = True
                    break
            if not dominated:
                pareto_points.append(point)

        return np.array(pareto_points)

    def draw(self, generation):
        """
        显示图像，不重新计算目标函数值
        """
        if self.visual_mode == 1:
            self.ax.scatter(self.F1, self.F2, s=10, c='lightgray', label='解')
            if self.pareto_points is not None:
                self.ax.scatter(self.pareto_points[:, 0], self.pareto_points[:, 1], c='#a94826', label='最优边界')
            self.ax.set_xlabel('f1')
            self.ax.set_ylabel('f2')
            self.ax.set_title(f'目标函数解空间 (第{generation + 1}代)')
            self.ax.legend()
            plt.draw()
            print("[ObjectiveVisualizer]重绘目标函数解空间")
        elif self.visual_mode == 2:
            pass

    def draw_individuals(self, individuals, generation, pause_time=0.2):
        """
        显示种群的个体点，不按rank。
                参数：
                    individuals (list): 种群对象个体列表，需要有objectives属性，格式为 [f1, f2]。
                    generation (int): 当前代数
        """
        if len(individuals[0].objectives) != 2:
            print("[ObjectiveVisualizer]画图只支持 2 目标优化问题")
            return
        self.ax.clear()  # 清除当前内容而不重新创建figure
        self.draw(generation)  # 绘制解空间
        f1_values = [ind.objectives[0] for ind in individuals]
        f2_values = [ind.objectives[1] for ind in individuals]
        plt.scatter(f1_values, f2_values, color='green', alpha=0.6, label=f'第{generation + 1}代种群个体')
        plt.legend()
        plt.pause(pause_time)
        if self.save_gif:
            self.gif_generator.add_frame(self.fig)

    def draw_individuals_by_rank(self, individuals, generation, pause_time=0.2):
        """
        按照rank显示种群的个体点
                参数：
                    individuals (list[list]): 种群对象个体按rank分组列表，需要有objectives属性，属性格式为 [f1, f2]。
                    分组代表rank，格式为 [ [ind1, ind2], [ind3, ind4],... ]。靠前的rank代表更优秀的个体。
                    generation (int): 当前代数
        """
        if len(individuals[0][0].objectives) != 2:
            print("[ObjectiveVisualizer]画图只支持 2 目标优化问题")
            return
        if self.visual_mode == 1:
            self.ax.clear()  # 清除当前内容而不重新创建figure
            self.draw(generation)  # 绘制解空间
            num_ranks = len(individuals)  # 计算rank数量
            colors = cm.get_cmap("plasma", num_ranks)

            for rank, group in enumerate(individuals):
                f1_values = [ind.objectives[0] for ind in group]
                f2_values = [ind.objectives[1] for ind in group]

                # 获取颜色映射，rank越小颜色越深
                rank_color = colors(rank / (num_ranks - 1))  # 归一化rank用于颜色映射
                plt.scatter(f1_values, f2_values, color=rank_color, alpha=0.8, label=f'rank:{rank}')

            plt.legend()
            plt.pause(pause_time)
            if self.save_gif:
                self.gif_generator.add_frame(self.fig)
        elif self.visual_mode == 2:
            print("未实现")
        else:
            print("未实现")

    def draw_populations(self, individuals, generation, pause_time=0.2):
        """
        显示种群的个体点,不按rank。
                参数：
                    individuals (list): 种群个体坐标列表，格式为元组 (f1, f2)，代表点的坐标
                    generation (int): 当前代数
        """
        if len(individuals[0]) != 2:
            print("[ObjectiveVisualizer]画图只支持 2 目标优化问题")
            return
        self.ax.clear()  # 清除当前内容而不重新创建figure
        self.draw(generation)  # 绘制解空间
        f1_values = [ind[0] for ind in individuals]
        f2_values = [ind[1] for ind in individuals]
        plt.scatter(f1_values, f2_values, color='green', alpha=0.6, label=f'第{generation + 1}代种群个体')
        plt.legend()
        plt.pause(pause_time)
        if self.save_gif:
            self.gif_generator.add_frame(self.fig)

    def draw_populations_by_rank(self, individuals, generation, ranks, max_rank, pause_time=0.2
                                 ):
        """
        按照rank显示种群的个体点
                参数：
                    individuals (list[list]): 种群个体坐标按rank分组列表，格式为元组 (f1, f2)，代表点的坐标。
                    分组代表rank，格式为 [ [ind1, ind2], [ind3, ind4],... ]。靠前的rank代表更优秀的个体。
                    generation (int): 当前代数
        """
        if len(individuals[0][0]) != 2:
            print("[ObjectiveVisualizer]画图只支持 2 目标优化问题")
            return
        self.ax.clear()  # 清除当前内容而不重新创建figure
        self.draw(generation)  # 绘制解空间
        print("未实现")  # TODO
        pass

    def save(self):
        """
        保存gif文件
        """
        if self.save_gif:
            self.gif_generator.save_gif()

    def close(self):
        """
        关闭figure
        """
        plt.close(self.fig)
