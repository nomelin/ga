import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from lib.tool import GIFGenerator

"""
ObjectiveVisualizer类用于可视化目标函数的解空间。
"""


class ObjectiveVisualizer:
    def __init__(self, variable_ranges, funcs=None, resolution=500, show_pareto=False, objectives=None, figsize=(8, 8),
                 visual_mode=1, save_gif=False, gif_name='default', queue=None):
        """
        初始化可视化工具，计算目标函数值并生成网格。

        参数:
            funcs (list)（如果这里没传，则需要在外部调用reCalculate函数）: 包含两个目标函数的列表，每个函数应接受相同数量的自变量，返回标量目标值。
            variable_ranges (list of tuples): 每个自变量的取值范围列表，格式为 [(x1_min, x1_max), (x2_min, x2_max)]。
            resolution (int): 采样分辨率，表示在每个自变量的范围内采样的点数。
            show_pareto (bool): 是否显示Pareto最优边界，默认为False。
            objectives (dict): 目标优化方向配置字典，例如 {'f1': 'min', 'f2': 'max'}。
            visual_mode 1:使用matplotlib;

        功能:
            初始化时计算目标函数值和Pareto前沿，并保存这些数据以供后续使用。
        """
        self.variable_ranges = variable_ranges
        self.resolution = resolution
        self.show_pareto = show_pareto
        self.objectives = objectives
        self.grids = self.generate_grid(variable_ranges, resolution)
        if funcs is not None:
            self.reCalculate(funcs)
        self.pareto_points = None
        if save_gif:
            self.gif_generator = GIFGenerator(gif_name + ".gif")
        self.visual_mode = visual_mode
        self.save_gif = save_gif

        if show_pareto and objectives and funcs is not None:
            print("[ObjectiveVisualizer]准备计算Pareto前沿")
            self.pareto_points = self.pareto_front(self.F1, self.F2, objectives)
            print("[ObjectiveVisualizer]Pareto前沿点数：", len(self.pareto_points))

        # 初始化figure和axes
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.t = None  # 初始化时间变量
        self.queue = queue  # 队列用于存储可视化数据

    def show_pareto_front(self):
        """
        显示Pareto前沿
        """
        if self.show_pareto and self.pareto_points is not None:
            self.draw(-1)
            self.ax.scatter(self.pareto_points[:, 0], self.pareto_points[:, 1], c='#a94826', label='最优边界')
            self.ax.legend()
            plt.draw()
            print("[ObjectiveVisualizer]显示Pareto前沿")
            if self.save_gif:
                self.gif_generator.add_frame(self.fig)

    def reCalculate(self, funcs, t=None):
        """
        重新计算目标函数值，并保存这些数据以供后续使用。

        参数:
            funcs (list): 包含目标函数的列表，每个函数接受相同数量的自变量和一个可选的时间变量t。
            t (float): 当前的时间参数，用于动态函数。如果不是动态函数，则不需要传入。
        """
        self.funcs = funcs
        self.t = t  # 更新时间变量
        print("[ObjectiveVisualizer]准备计算目标函数值")
        if self.visual_mode == 1:
            self.F1, self.F2 = self.calculate_objective_values(funcs, self.grids, t)
        elif self.visual_mode == 2:
            self.points = self.calculate_objective_values(funcs, self.grids, t)  # 坐标点数组
        print(f"[ObjectiveVisualizer]目标函数值计算完成,t={t}")

    def generate_grid(self, variable_ranges, resolution):
        # 根据分辨率生成网格
        grids = [np.linspace(r[0], r[1], resolution) for r in variable_ranges]
        return np.meshgrid(*grids)

    def calculate_objective_values(self, funcs, grids, t=None):
        """
        计算所有目标函数的值。

        参数:
            funcs (list): 包含目标函数的列表。
            grids (list): 网格数据。
            t (float): 时间参数。

        返回:
            list: 每个目标函数的值。
        """
        if self.visual_mode == 1:
            if t is not None:
                return [func(*grids, t) for func in funcs]
            return [func(*grids) for func in funcs]
        elif self.visual_mode == 2:
            mesh_shape = grids[0].shape
            grid_points = np.vstack([g.ravel() for g in grids]).T  # 将网格点展平为二维数组
            if t is not None:
                objectives = np.array([func(*pt, t) for pt in grid_points for func in funcs])
            else:
                objectives = np.array([func(*pt) for pt in grid_points for func in funcs])

            # 将结果整形成二维坐标数组 [[f1, f2], ...]
            return objectives.reshape(-1, len(funcs))  # 每行是一个解点，每列对应一个目标函数值

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
            solution_points = self.points.tolist()  # 解空间点
            population_data = [
                {"rank": rank, "points": [{"f1": ind.objectives[0], "f2": ind.objectives[1]} for ind in group]}
                for rank, group in enumerate(individuals)
            ]

            # 推送数据到队列
            if self.queue:
                self.queue.put({
                    "generation": generation,
                    "solution_points": solution_points,
                    "population_data": population_data
                })
            print(f"[ObjectiveVisualizer] Generation {generation} 数据已推送到队列.")
        else:
            print("[ObjectiveVisualizer] 未实现的可视化模式")

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
