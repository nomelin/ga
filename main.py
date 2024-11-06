from funcs import POL
from ga.nsgaii import nsga2
from lib.visual import ObjectiveVisualizer


# 定义目标函数
def f1(x1, x2):
    return x1 ** 2 + x2 ** 2


def f2(x1, x2):
    return (x1 - 1) ** 2 + x2 ** 2


def f3(x1, x2):
    return (x1 + 1) ** 2 + x2


# 设置参数
variable_ranges = [(-2, 2), (-2, 2)]  # x1 和 x2 的取值范围

# 初始化可视化工具
visualizer = ObjectiveVisualizer(
    variable_ranges=POL.variable_ranges,
    # resolution=500,
    # show_pareto=True,
    # objectives={'f1': 'min', 'f2': 'min'}
    save_gif=True,  # 保存动画
    gif_name='nsga2-POL-140',
    figsize=(12, 12)
)

population = nsga2(
    funcs_dict={0: [POL.funcs, ['min', 'min']], 10: [POL.funcs, ['min', 'max']], 20: [POL.funcs, ['max', 'max']]},
    variable_ranges=POL.variable_ranges,
    precision=0.01,
    pop_size=140,
    num_generations=30,
    visualizer=visualizer,
    crossover_rate=0.9,
    mutation_rate=0.1
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
