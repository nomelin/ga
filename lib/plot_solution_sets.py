import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def plot_solution_sets(current_solutions, predicted_solutions, file_name="solution_sets.gif"):
    """
    绘制两个解集的可视化图，并将其保存为GIF格式。

    参数:
    - current_solutions (list of list): 当前解集，每个解是一个列表 [x1, x2, ..., xn]。
    - predicted_solutions (list of list): 预测解集，每个解是一个列表 [x1, x2, ..., xn]。
    - file_name (str): 保存的GIF文件名，默认为 "solution_sets.gif"。
    """
    # 获取当前目录的上级目录
    parent_dir = os.path.dirname(os.getcwd())

    # 创建平行的image文件夹
    image_dir = os.path.join(parent_dir, 'image')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(8, 6))

    # 将解集转换为可绘制的坐标
    current_x = [solution[0] for solution in current_solutions]
    current_y = [solution[1] for solution in current_solutions]

    predicted_x = [solution[0] for solution in predicted_solutions]
    predicted_y = [solution[1] for solution in predicted_solutions]

    # 绘制散点图
    scatter_current = ax.scatter(current_x, current_y, color='blue', label='Current Solutions')
    scatter_predicted = ax.scatter(predicted_x, predicted_y, color='red', label='Predicted Solutions')

    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.set_title('Comparison of Current and Predicted Solution Sets')
    ax.legend()
    ax.grid(True)

    # 动画更新函数
    def update(frame):
        # 每一帧更新图像
        ax.clear()
        ax.scatter(current_x, current_y, color='blue', label='Current Solutions')
        ax.scatter(predicted_x[:frame], predicted_y[:frame], color='red', label='Predicted Solutions')
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_title('Comparison of Current and Predicted Solution Sets')
        ax.legend()
        ax.grid(True)

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(predicted_solutions), repeat=False)

    # 保存为GIF（路径为当前目录的上级目录中的image文件夹）
    gif_path = os.path.join(image_dir, file_name)
    ani.save(gif_path, writer='pillow', fps=2)

    plt.show()


# 示例用法
current_solutions = [[1, 2], [3, 1], [2, 3], [4, 2]]
predicted_solutions = [[1.2, 2.1], [3.1, 0.9], [2.2, 3.2], [4.1, 2.2]]


# 传入自定义文件名
plot_solution_sets(current_solutions, predicted_solutions, "custom_solution_sets.gif")
