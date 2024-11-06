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
pop_size = 50  # 种群大小
num_generations = 15  # 迭代次数

# 初始化可视化工具
visualizer = ObjectiveVisualizer(
    funcs=[f1, f2],
    variable_ranges=variable_ranges,
    # resolution=500,
    # show_pareto=True,
    # objectives={'f1': 'min', 'f2': 'min'}
    save_gif=True,  # 保存动画
    gif_name='nsga2-test'
)

population = nsga2(
    funcs_dict={0: [[f1, f2], ['min', 'min']], 6: [[f1, f2], ['min', 'max']]},
    variable_ranges=variable_ranges,
    precision=precision,
    pop_size=pop_size,
    num_generations=num_generations,
    visualizer=visualizer,
    crossover_rate=0.9,
    mutation_rate=0.2
)

visualizer.save()  # 保存动画

# visualizer2 = ObjectiveVisualizer(
#     funcs=[f1, f2],
#     variable_ranges=variable_ranges,
# )
#
# nsga2(
#     funcs=[f1, f2],
#     variable_ranges=variable_ranges,
#     precision=precision,
#     pop_size=pop_size,
#     num_generations=num_generations,
#     visualizer=visualizer2
# )

objective_values = [[f1(x1, x2), f2(x1, x2)] for x1, x2 in population]

# 输出结果
print("最终种群（解码后）：")
for vars, obj_vals in zip(population, objective_values):
    print(f"变量值: {vars}, 目标值: {obj_vals}")

input("Press any key to exit...")
