import numpy as np
from matplotlib import pyplot as plt

from lib.tool import GIFGenerator
from lib.visual import ObjectiveVisualizer


# 定义目标函数
def f1(x1, x2):
    return x1 ** 2 + x2 ** 2


def f2(x1, x2):
    return (x1 - 2) ** 2 + x2 ** 2


# 设置变量范围和分辨率
ranges = [(-2, 2), (-2, 2)]
resolution = 500

plt.ion()  # 开启interactive mode交互模式

# 初始化可视化工具
visualizer = ObjectiveVisualizer(
    funcs=[f1, f2],
    variable_ranges=ranges,
    resolution=resolution,
    # show_pareto=True,
    # objectives={'f1': 'min', 'f2': 'max'}
)
gif_gen = GIFGenerator(filename='save/dynamic_population.gif')

# 动态展示
for generation in range(10):
    visualizer.draw()  # 绘制解空间

    # 生成或更新个体位置
    individuals = [(np.random.uniform(-2, 2), np.random.uniform(-2, 2)) for _ in range(3)]
    f1_values = [f1(x1, x2) for x1, x2 in individuals]
    f2_values = [f2(x1, x2) for x1, x2 in individuals]

    # 在同一个figure上绘制个体点，使用绿色
    plt.scatter(f1_values, f2_values, color='green', alpha=0.6, label=f'第{generation + 1}代种群个体')
    plt.legend()

    gif_gen.add_frame(plt.gcf())  # 添加当前帧到GIF
    plt.pause(0.2)

plt.ioff()  # 关闭交互模式
gif_gen.save_gif()  # 保存GIF文件
plt.show()
