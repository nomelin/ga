
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

class LSTMPopulationPredictor:
    def __init__(self, input_dim, output_dim, window_size=3, population_size=100):
        # 初始化 LSTM 种群预测器的参数
        self.input_dim = input_dim  # 输入维度，即每个解的变量数量
        self.output_dim = output_dim  # 输出维度，与输入维度一致
        self.window_size = window_size  # 时间窗口大小，决定 LSTM 输入的时间步数
        self.population_size = population_size  # 种群规模，即每次预测生成的解的数量
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # 用于归一化数据
        self.model = self._build_lstm_model()  # 构建 LSTM 模型

    def _build_lstm_model(self):
        # 构建 LSTM 神经网络模型
        model = Sequential([
            Input(shape=(self.window_size, self.input_dim)),  # 输入层，指定时间窗口和特征维度
            LSTM(50, activation='relu', return_sequences=False),  # LSTM 层，50 个神经元
            Dense(self.output_dim * self.population_size)  # 输出层，输出为 population_size 个解
        ])
        model.compile(optimizer='adam', loss='mse')  # 编译模型，使用均方误差作为损失函数
        return model

    def train(self, historical_data, target_data, epochs=10, batch_size=16):
        # 训练 LSTM 模型
        if len(historical_data) < self.window_size or len(target_data) < self.window_size:
            return  # 如果历史数据或目标数据不足以构成一个完整的时间窗口，则不训练

        # 对输入和目标数据进行归一化处理
        historical_data = self.scaler.fit_transform(historical_data.reshape(-1, historical_data.shape[-1])).reshape(
            historical_data.shape)
        target_data = self.scaler.fit_transform(target_data.reshape(-1, target_data.shape[-1])).reshape(
            target_data.shape)

        # 构建时间序列训练数据集
        x_train, y_train = [], []
        for i in range(len(historical_data) - self.window_size):
            x_train.append(historical_data[i:i + self.window_size])  # 输入：时间窗口内的数据
            if i + self.window_size < len(target_data):  # 确保索引不会超出 target_data 的范围
                y_train.append(target_data[i + self.window_size])  # 输出：时间窗口结束后的目标数据

        # 转换为 NumPy 数组
        x_train, y_train = np.array(x_train), np.array(y_train)

        # 确保输入数据维度正确 (三维)
        if x_train.ndim == 2:
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], -1)

        # 确保输出数据维度正确 (二维)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        # 将输出数据重复以匹配种群规模
        y_train = np.repeat(y_train, self.population_size, axis=1).reshape(-1, self.output_dim * self.population_size)

        # 使用处理好的数据训练 LSTM 模型
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict_population(self, recent_data, current_pop_size):
        """
        使用训练好的 LSTM 模型预测新的种群。
        :param recent_data: 最近的时间窗口数据，形状为 (window_size, input_dim)
        :param current_pop_size: 当前种群的数量
        :return: 预测的种群，形状为 (current_pop_size, output_dim)
        """
        if not hasattr(self.scaler, 'scale_'):
            raise ValueError("MinMaxScaler 尚未进行拟合，请先调用 'train' 方法训练模型。")

        if len(recent_data) < self.window_size:
            return None  # 如果输入数据不足以构成完整的时间窗口，无法进行预测

        # 对输入数据进行归一化
        recent_data = self.scaler.transform(recent_data.reshape(-1, recent_data.shape[-1])).reshape(recent_data.shape)

        # 使用 LSTM 进行预测
        prediction = self.model.predict(np.expand_dims(recent_data, axis=0))
        prediction = prediction.reshape(self.population_size, self.output_dim)

        # 将预测结果反归一化
        predicted_population = self.scaler.inverse_transform(prediction)

        # 调整预测结果以匹配当前种群数量
        if self.population_size > current_pop_size:
            # 如果预测的种群数量大于当前种群数量，截取前 current_pop_size 个解
            predicted_population = predicted_population[:current_pop_size]
        elif self.population_size < current_pop_size:
            # 如果预测的种群数量小于当前种群数量，重复预测结果以匹配当前种群数量
            repeat_times = current_pop_size // self.population_size + 1
            predicted_population = np.tile(predicted_population, (repeat_times, 1))[:current_pop_size]

        return predicted_population
    def enforce_boundaries(self, population, lower_bound, upper_bound):
        # 将解强制限制在边界范围内
        return np.clip(population, lower_bound, upper_bound)

    def add_random_noise(self, population, noise_level=0.01):
        # 为种群解添加随机噪声，以增加解的多样性
        noise = np.random.uniform(-noise_level, noise_level, population.shape)
        return population + noise

# ==== 使用示例 ====
if __name__ == "__main__":
    input_dim = 5
    output_dim = 5
    window_size = 3
    population_size = 100
    num_samples = 200

    # 模拟历史数据
    historical_data = np.random.rand(num_samples, input_dim)
    print("Historical Data Shape:", historical_data.shape)
    print("Historical Data:", historical_data)
    # 模拟目标数据
    target_data = np.random.rand(num_samples, output_dim)
    print("Target Data Shape:", target_data.shape)
    print("Target Data:", target_data)

    # 初始化模型
    model = LSTMPopulationPredictor(input_dim, output_dim, window_size, population_size)
    model.train(historical_data, target_data, epochs=10, batch_size=16)

    # 最近的时间窗口数据
    recent_data = np.random.rand(window_size, input_dim)
    predicted_population = model.predict_population(recent_data,100)

    # 进行边界处理和多样性增强
    lower_bound = np.zeros(output_dim)
    upper_bound = np.ones(output_dim)
    predicted_population = model.enforce_boundaries(predicted_population, lower_bound, upper_bound)
    predicted_population = model.add_random_noise(predicted_population)
    print(predicted_population.dtype)

    print("Predicted Population Shape:", predicted_population.shape)
    print("Predicted Population:", predicted_population)
