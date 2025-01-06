import matplotlib.pyplot as plt


def plot_solution_sets(current_solutions, predicted_solutions):
    """
    绘制两个解集的可视化图。

    参数:
    - current_solutions (list of list): 当前解集，每个解是一个列表 [x1, x2, ..., xn]。
    - predicted_solutions (list of list): 预测解集，每个解是一个列表 [x1, x2, ..., xn]。
    """
    # 将解集转换为可绘制的坐标
    current_x = [solution[0] for solution in current_solutions]
    current_y = [solution[1] for solution in current_solutions]

    predicted_x = [solution[0] for solution in predicted_solutions]
    predicted_y = [solution[1] for solution in predicted_solutions]

    plt.figure(figsize=(8, 6))
    plt.scatter(current_x, current_y, color='blue', label='Current Solutions')
    plt.scatter(predicted_x, predicted_y, color='red', label='Predicted Solutions')

    # 可选：为每个点添加标签（若需要）
    # for i, coord in enumerate(current_solutions):
    #     plt.text(coord[0], coord[1], f"{i}", fontsize=9)

    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Comparison of Current and Predicted Solution Sets')
    plt.legend()
    plt.grid(True)
    plt.show()


# 示例用法
# current_solutions = [[1, 2], [3, 1], [2, 3], [4, 2]]
# predicted_solutions = [[1.2, 2.1], [3.1, 0.9], [2.2, 3.2], [4.1, 2.2]]
# plot_solution_sets(current_solutions, predicted_solutions)
