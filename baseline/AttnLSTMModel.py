import os
import random
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
warnings.filterwarnings("ignore")


def fill_missing(values):
    if np.any(np.isnan(values)):
        values = pd.Series(values.flatten()).interpolate().values.reshape(-1, 1)
    return values


def create_dataset_single(target, time_step=10):
    X, y = [], []
    for i in range(len(target) - time_step):
        X.append(target[i:i + time_step])
        y.append(target[i + time_step, 0])
    return np.array(X), np.array(y)


def train_test_split(X, y, train_ratio=0.8):
    train_size = int(len(X) * train_ratio)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]


def load_target_series(dataset_directory, target_filename, num_rows=10000):
    filepath = os.path.join(dataset_directory, target_filename)
    data = pd.read_csv(filepath, header=None).iloc[:num_rows]
    series = data[2].values[:num_rows].reshape(-1, 1)
    return fill_missing(series)


def train_model(model, X_train, y_train, num_epochs=400):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/400], Loss: {loss.item():.4f}")


def predict_model(model, X, scaler_target):
    model.eval()
    with torch.no_grad():
        preds = model(X)
        preds = scaler_target.inverse_transform(preds.numpy())
    return preds


def direction_accuracy(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    correct = sum(1 for t in range(1, len(y_true)) if (y_true[t] - y_true[t - 1]) * (y_pred[t] - y_pred[t - 1]) > 0)
    return correct / (len(y_true) - 1)


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    da = direction_accuracy(y_true, y_pred)
    return mse, rmse, mae, r2, da


# 带 Attention 的 LSTM 模型定义
class AttnLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=4, dropout=0.3):
        super(AttnLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(out), dim=1)
        out = torch.sum(attn_weights * out, dim=1)
        return self.fc(out)


def main():
    dataset_directory = "DataSet1"
    target_filename = "data.csv"
    time_step = 10
    series = load_target_series(dataset_directory, target_filename)
    X, y = create_dataset_single(series, time_step)
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled)
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    model = AttnLSTMModel(input_size=1)
    print("训练 带 Attention 的 LSTM 模型...")
    train_model(model, X_train, y_train, num_epochs=400)
    preds = predict_model(model, X_test, scaler)
    y_test_actual = scaler.inverse_transform(y_test.numpy())
    metrics = calculate_metrics(y_test_actual, preds)
    print("AttnLSTM 测试指标 (MSE, RMSE, MAE, R2, DA):", metrics)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, label="真实值")
    plt.plot(preds, label="预测值")
    plt.title("AttnLSTM 模型测试集预测结果")
    plt.xlabel("样本索引")
    plt.ylabel("值")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
