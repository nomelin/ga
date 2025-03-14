import json

import torch
from matplotlib import pyplot as plt

from model.PopulationPredictorLSTM import PopulationPredictorLSTM, save_model, load_model
from model.loss import calculate_loss


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
    print(f"Loaded {len(samples)} samples from {json_file}.")
    return samples


def train_model_no_batch(model, samples, epochs, alpha=1.0, beta=0.1, lr=1e-3, device='cpu'):
    """
    不使用批量处理的 LSTM 模型训练函数。
    参数:
        model (nn.Module): 预测模型。
        samples (list): 数据样本列表，每个样本包含输入和目标种群。
        epochs (int): 训练轮数。
        alpha (float): 匹配损失权重。
        beta (float): 多样性损失权重。
        lr (float): 学习率。
        device (str): 运行设备（'cpu' 或 'cuda'）。
    """
    model.to(device)
    print(f"Training on {device}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(epochs):
        total_loss = 0.0

        for sample in samples:
            inputs, target = sample
            inputs, target = inputs.to(device), target.to(device)  # 添加 batch 维度
            # print(f"size of inputs: {inputs.size()}, size of target: {target.size()}")

            optimizer.zero_grad()

            # 前向传播
            predicted = model(inputs)
            # print(f"size of predicted: {predicted.size()}")

            # 计算损失
            loss = calculate_loss(predicted, target, alpha, beta)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(samples)
        loss_history.append(avg_loss)
        print(f"----- Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f} -----")
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_curve.png")
    plt.show()


if __name__ == '__main__':
    json_file = '../data/optimal_solutions_dy3.json'
    epochs = 100
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据
    samples = load_population_data(json_file)
    weight_path = "weight/model.pth"

    # 初始化模型
    model = PopulationPredictorLSTM()

    if input("是否加载模型? (y/n) ") == 'y':
        load_model(model, weight_path, device)

    train_model_no_batch(model, samples, epochs, lr=learning_rate, device=device)

    print("Training finished.")
    if input("是否保存模型? (y/n) ") == 'y':
        save_model(model, weight_path)
