import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

class LSTMPopulationPredictor:
    def __init__(self, input_dim, output_dim, window_size=3, population_size=100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.population_size = population_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_lstm_model()

    def _build_lstm_model(self):
        model = Sequential()
        model.add(Input(shape=(self.window_size, self.input_dim)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(self.output_dim * self.population_size))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, historical_data, target_data, epochs=10, batch_size=16):
        if len(historical_data) < self.window_size:
            raise ValueError("历史数据长度不足以形成完整的时间窗口")

        # 数据归一化
        historical_data = self.scaler.fit_transform(historical_data.reshape(-1, self.input_dim)).reshape(historical_data.shape)
        target_data = self.scaler.fit_transform(target_data.reshape(-1, self.output_dim)).reshape(target_data.shape)

        # 构建时间窗口训练数据
        x_train, y_train = [], []
        for i in range(len(historical_data) - self.window_size):
            x_train.append(historical_data[i:i + self.window_size])

            # ✅ 修复：将 target_data 扩展为 population_size * output_dim
            y_target_expanded = np.tile(target_data[i + self.window_size], (self.population_size, 1))
            y_train.append(y_target_expanded.flatten())

        x_train, y_train = np.array(x_train), np.array(y_train)

        # ✅ 确保维度正确
        if x_train.ndim != 3:
            raise ValueError(f"x_train 必须是三维数组，但得到了 {x_train.shape}")
        if y_train.ndim != 2 or y_train.shape[1] != self.output_dim * self.population_size:
            raise ValueError(f"y_train 必须是二维数组，形状为 (samples, {self.output_dim * self.population_size})，但得到了 {y_train.shape}")

        # 训练模型
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict_population(self, recent_data):
        if len(recent_data) < self.window_size:
            raise ValueError("最近数据长度不足")

        # 数据归一化
        recent_data = self.scaler.transform(recent_data.reshape(-1, self.input_dim)).reshape(1, self.window_size, self.input_dim)
        prediction = self.model.predict(recent_data)
        prediction = self.scaler.inverse_transform(prediction.reshape(self.population_size, self.output_dim))
        return prediction

# ==== 使用示例 ====
if __name__ == "__main__":
    input_dim = 5
    output_dim = 5
    window_size = 3
    population_size = 100
    num_samples = 200

    # 模拟数据
    historical_data = np.random.rand(num_samples, input_dim)
    target_data = np.random.rand(num_samples, output_dim)

    # 初始化和训练
    model = LSTMPopulationPredictor(input_dim, output_dim, window_size, population_size)
    model.train(historical_data, target_data, epochs=10, batch_size=16)

    # 预测
    recent_data = np.random.rand(window_size, input_dim)
    predicted_population = model.predict_population(recent_data)
    print("Predicted Population Shape:", predicted_population.shape)
    print("Predicted Population:", predicted_population)
