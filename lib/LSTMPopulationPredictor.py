import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


class LSTMPopulationPredictor:
    def __init__(self, input_dim, output_dim, window_size=3, population_size=100):
        """
        LSTM种群预测器，用于动态环境下的多目标优化。
        :param input_dim: 输入特征维度
        :param output_dim: 输出特征维度
        :param window_size: 时间窗口大小
        :param population_size: 每次预测的解的数量（种群规模）
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.population_size = population_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_lstm_model()

    def _build_lstm_model(self):
        """构建LSTM模型，直接输出多个解"""
        model = Sequential()
        model.add(Input(shape=(self.window_size, self.input_dim)))
        model.add(LSTM(50, activation='relu', return_sequences=False))
        model.add(Dense(self.output_dim * self.population_size))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, historical_data, target_data, epochs=10, batch_size=16):
        """训练LSTM模型"""
        if len(historical_data) < self.window_size:
            return  # 数据不足时不训练

        # 数据归一化
        historical_data = self.scaler.fit_transform(historical_data.reshape(-1, historical_data.shape[-1])).reshape(
            historical_data.shape)
        target_data = self.scaler.fit_transform(target_data.reshape(-1, target_data.shape[-1])).reshape(
            target_data.shape)

        # 构建时间序列训练集
        x_train, y_train = [], []
        for i in range(len(historical_data) - self.window_size):
            x_train.append(historical_data[i:i + self.window_size])
            y_train.append(target_data[i + self.window_size])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # 确保 x_train 是三维数组
        if x_train.ndim == 2:
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],
                                      -1)  # 转换为 (num_samples, window_size, input_dim)

        # 确保 y_train 是二维数组
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)  # 如果是一维数组，转换为二维数组

        # 重复 y_train 以匹配种群规模
        y_train = np.repeat(y_train, self.population_size, axis=1).reshape(-1, self.output_dim * self.population_size)

        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict_population(self, recent_data):
        """预测多个解的种群"""
        if not hasattr(self.scaler, 'scale_'):  # 检查是否已拟合
            raise ValueError("MinMaxScaler is not fitted yet. Call 'train' method first.")

        if len(recent_data) < self.window_size:
            return None  # 数据不足时不预测

        recent_data = self.scaler.transform(recent_data.reshape(-1, recent_data.shape[-1])).reshape(recent_data.shape)
        prediction = self.model.predict(np.expand_dims(recent_data, axis=0))
        prediction = prediction.reshape(self.population_size, self.output_dim)
        return self.scaler.inverse_transform(prediction)

    def enforce_boundaries(self, population, lower_bound, upper_bound):
        """边界约束处理"""
        return np.clip(population, lower_bound, upper_bound)

    def add_random_noise(self, population, noise_level=0.01):
        """多样性增强：添加随机噪声"""
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
    target_data = np.random.rand(num_samples, output_dim)

    # 初始化模型
    model = LSTMPopulationPredictor(input_dim, output_dim, window_size, population_size)
    model.train(historical_data, target_data, epochs=10, batch_size=16)

    # 最近的时间窗口数据
    recent_data = np.random.rand(window_size, input_dim)
    predicted_population = model.predict_population(recent_data)

    # 进行边界处理和多样性增强
    lower_bound = np.zeros(output_dim)
    upper_bound = np.ones(output_dim)
    predicted_population = model.enforce_boundaries(predicted_population, lower_bound, upper_bound)
    predicted_population = model.add_random_noise(predicted_population)

    print("Predicted Population Shape:", predicted_population.shape)
    print("Predicted Population:", predicted_population)
