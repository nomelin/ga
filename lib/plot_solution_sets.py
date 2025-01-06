import os

import imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt


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


def savetogif(current, predicted, file_name="solution_sets.gif"):
    """
    Save a sequence of current and predicted populations as a GIF.

    Args:
        current (list): A list of populations (each a list of individuals, each a 2D coordinate) representing the current state.
        predicted (list): A list of populations (same format as current) representing the predicted state.
        file_name (str): The name of the output GIF file. Default is "solution_sets.gif".
    """
    if len(current) != len(predicted):
        raise ValueError("current and predicted must have the same length.")

    # Directory to save temporary images
    temp_dir = "temp_gif_frames"
    os.makedirs(temp_dir, exist_ok=True)

    temp_files = []

    for t, (cur_pop, pred_pop) in enumerate(zip(current, predicted)):
        print("-", end="", flush=True)
        plt.figure(figsize=(6, 6))

        # Plot current population
        cur_pop = zip(*cur_pop)  # Transpose for easier plotting
        plt.scatter(*cur_pop, color='blue', label='Current', alpha=0.7)

        # Plot predicted population
        pred_pop = zip(*pred_pop)  # Transpose for easier plotting
        plt.scatter(*pred_pop, color='red', label='Predicted', alpha=0.7)

        plt.title(f"Population Evolution - Step {t + 1}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)

        # Save frame as temporary image
        frame_path = os.path.join(temp_dir, f"frame_{t:03d}.png")
        plt.savefig(frame_path)
        temp_files.append(frame_path)
        plt.close()

    # Create GIF
    with imageio.get_writer(file_name, mode='I', duration=0.5) as writer:
        for file_path in temp_files:
            image = imageio.imread(file_path)
            writer.append_data(image)

    # Clean up temporary files
    for file_path in temp_files:
        os.remove(file_path)

    os.rmdir(temp_dir)
    print(f"GIF saved as {file_name}")

