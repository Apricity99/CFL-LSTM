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

# 设置随机种子，确保实验结果可复现
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")


##########################################
# 数据加载与预处理函数
##########################################

def fill_missing(values):
    if np.any(np.isnan(values)):
        values = pd.Series(values.flatten()).interpolate().values.reshape(-1, 1)
    return values


def create_dataset_single(target, time_step=10):
    X, y = [], []
    L = len(target)
    for i in range(L - time_step):
        X.append(target[i:i + time_step])
        y.append(target[i + time_step, 0])
    return np.array(X), np.array(y)


def train_test_split(X, y, train_ratio=0.8):
    train_size = int(len(X) * train_ratio)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]


def load_target_series(dataset_directory, target_filename, num_rows=10000):
    target_file = os.path.join(dataset_directory, target_filename)
    data = pd.read_csv(target_file, header=None).iloc[:num_rows]
    series = data[2].values[:num_rows].reshape(-1, 1)
    return fill_missing(series)


##########################################
# GRU 模型定义
##########################################

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


##########################################
# 训练、预测与评价函数
##########################################

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


##########################################
# main 函数
##########################################

def main():
    # 数据集路径与目标文件名称（请根据实际情况修改）
    dataset_directory = "DataSet1"
    target_filename = "data.csv"
    time_step = 10

    # 加载目标序列数据
    series = load_target_series(dataset_directory, target_filename)
    X, y = create_dataset_single(series, time_step)
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled)

    # 转为 torch 张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # 构造 GRU 模型
    model = GRUModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.3)
    print("训练 GRU 模型...")
    train_model(model, X_train, y_train, num_epochs=400)

    # 模型预测
    preds = predict_model(model, X_test, scaler)
    y_test_actual = scaler.inverse_transform(y_test.numpy())
    metrics = calculate_metrics(y_test_actual, preds)
    print("GRU 测试指标 (MSE, RMSE, MAE, R2, DA):", metrics)

    # 绘制预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, label="真实值")
    plt.plot(preds, label="预测值")
    plt.title("GRU 模型测试集预测结果")
    plt.xlabel("样本索引")
    plt.ylabel("值")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
