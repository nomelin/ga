from ga.nsgaii import nsga2
from lib.visual import ObjectiveVisualizer


# 定义目标函数
def f1(x1, x2):
    return x1 ** 2 + x2 ** 2


def f2(x1, x2):
    return (x1 - 1) ** 2 + x2 ** 2


# 设置参数
variable_ranges = [(-2, 2), (-2, 2)]  # x1 和 x2 的取值范围
precision = 0.01  # 期望的搜索精度
pop_size = 100  # 种群大小
num_generations = 10  # 迭代次数

# 初始化可视化工具
visualizer = ObjectiveVisualizer(
    funcs=[f1, f2],
    variable_ranges=variable_ranges,
    resolution=500,
    # show_pareto=True,
    # objectives={'f1': 'min', 'f2': 'max'}
)

# 运行 NSGA-II 算法
population = nsga2(
    funcs=[f1, f2],
    variable_ranges=variable_ranges,
    precision=precision,
    pop_size=pop_size,
    num_generations=num_generations,
    visualizer=visualizer
)

objective_values = [[f1(x1, x2), f2(x1, x2)] for x1, x2 in population]

# 输出结果
print("最终种群（解码后）：")
for vars, obj_vals in zip(population, objective_values):
    print(f"变量值: {vars}, 目标值: {obj_vals}")

input("Press any key to exit...")
