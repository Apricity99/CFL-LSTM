import torch
import torch.nn as nn

class CFLLSTMWeightedModel(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=1, dropout=0.5, rho_threshold=0.3):
        super().__init__()
        self.input_size = input_size
        self.rho_threshold = rho_threshold
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, rho_weights=None):
        """
        x: Tensor [B, T, input_size]
        rho_weights: Tensor [input_size], 一般由 FCC 分析模块提供
        """
        if rho_weights is not None:
            mask = rho_weights >= self.rho_threshold
            if mask.sum() == 0:
                raise ValueError("所有辅助序列的 rho 都小于阈值，无法构建模型输入")
            x = x.clone()
            x[:, :, ~mask] = 0.0
        x = self.relu(self.conv1(x.transpose(1, 2)))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]) 
