import json

import torch


def load_population_data(json_file):
    """
    从 JSON 文件中读取种群数据。
    参数:
        json_file (str): JSON 文件路径。
    返回:
        List[Tuple[Tensor, Tensor]]: 数据样本列表，每个样本是 (inputs, target) 的元组。
            - inputs: (3, 100, 2) 的 Tensor，前 3 个时刻的种群数据。
            - target: (100, 2) 的 Tensor，目标种群数据。
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    samples = []
    for sample in data:
        inputs = torch.tensor(sample[:3], dtype=torch.float32)  # 前 3 个时刻种群
        target = torch.tensor(sample[3], dtype=torch.float32)  # 目标种群
        samples.append((inputs, target))
    return samples

if __name__ == '__main__':
    # 测试 load_population_data 函数
    json_file = 'data/population.json'
