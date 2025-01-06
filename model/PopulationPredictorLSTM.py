import torch.nn as nn


class PopulationPredictorLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, output_dim=2):
        """
        LSTM 模型，用于预测下一个种群位置。
        参数:
            input_dim (int): 输入特征维度（每个个体的坐标维度）。
            hidden_dim (int): LSTM 隐藏层维度。
            num_layers (int): LSTM 层数。
            output_dim (int): 输出特征维度（每个个体的坐标维度）。
        """
        super(PopulationPredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        """
        前向传播。
        参数:
            x (Tensor): 输入数据，形状为 (time_steps, num_individuals, input_dim)。
        返回:
            Tensor: 预测的种群数据，形状为 (num_individuals, output_dim)。
        """
        time_steps, num_individuals, input_dim = x.size()
        x = x.view(num_individuals, time_steps, input_dim)  # 调整维度为 (num_individuals, time_steps, input_dim)
        lstm_out, _ = self.lstm(x)  # (num_individuals, time_steps, hidden_dim)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        output = self.fc(lstm_out)  # (num_individuals, output_dim)
        return output
