import numpy as np
from PIL import Image


class GIFGenerator:
    def __init__(self, filename='animation.gif', fps=10):
        """
        初始化GIF生成器。

        参数:
            filename (str): 保存的GIF文件名。
            fps (int): 每秒帧数。
        """
        self.filename = filename
        self.fps = fps
        self.frames = []  # 用于保存每帧图像

    def add_frame(self, figure):
        """
        将当前figure添加为一帧。

        参数:
            figure (matplotlib.figure.Figure): 当前绘制的figure对象。
        """
        # 使用Pillow将figure保存为图像帧
        figure.canvas.draw()  # TODO: 此行代码在cmd中运行缓慢，需要优化。与figsize无关。
        image = np.array(figure.canvas.renderer.buffer_rgba())
        self.frames.append(Image.fromarray(image))
        print(f"[GIF] 添加一帧，总帧数 {len(self.frames)}")

    def save_gif(self):
        """
        将保存的帧组合成GIF文件。
        """
        if self.frames:
            print(f"[GIF] 开始保存动画，总帧数 {len(self.frames)}")
            self.frames[0].save(
                self.filename, save_all=True, append_images=self.frames[1:], duration=1000 / self.fps, loop=0
            )
            print(f"[GIF] 动画已保存为 {self.filename}")
        else:
            print("[GIF] 没有帧可保存")
