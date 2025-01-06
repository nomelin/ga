import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os


def plot_solution_sets_gif(solution_sets, filename="solution_sets.gif", folder="../image"):
    """
    绘制解集的动态变化过程，并保存为 GIF 文件到当前目录下的 image 文件夹。

    参数:
    - solution_sets (list of list): 每个时间步的解集，每个解集是一个列表 [x1, x2, ..., xn]。
    - filename (str): 保存的 GIF 文件名。
    - folder (str): 保存 GIF 的文件夹路径，默认为当前目录下的 "image"。
    """
    # 确保目标文件夹存在
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 完整的文件路径
    filepath = os.path.join(folder, filename)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 初始化散点图
    scatter = ax.scatter([], [], c='blue', label='Solutions')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.set_title('Dynamic Solution Sets Evolution')
    ax.legend()
    ax.grid(True)

    # 更新函数：用于每一帧更新散点图
    def update(frame):
        solutions = solution_sets[frame]
        x = [s[0] for s in solutions]
        y = [s[1] for s in solutions]
        scatter.set_offsets(np.c_[x, y])
        ax.set_title(f'Dynamic Solution Sets Evolution (Frame {frame + 1}/{len(solution_sets)})')

    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(solution_sets), repeat=True)

    # 保存为 GIF 文件
    ani.save(filepath, writer='pillow', fps=5)
    plt.close()
    print(f"[GIF 已保存] 文件保存为: {filepath}")


# ✅ 示例用法
if __name__ == "__main__":
    # 生成一些示例解集数据
    solution_sets = [
        np.random.uniform(-2, 2, (20, 2)).tolist() for _ in range(20)
    ]
    plot_solution_sets_gif(solution_sets, "dynamic_solutions.gif")
