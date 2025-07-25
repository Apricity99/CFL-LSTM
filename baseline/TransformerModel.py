import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim=32, seq_len=10, nhead=4, num_layers=2, hidden_dim=128):
        super(TransformerModel, self).__init__()
        assert input_dim % nhead == 0, "input_dim must be divisible by nhead"
        self.input_proj = nn.Linear(1, input_dim)  # 输入维度从1映射到input_dim
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(seq_len * input_dim, 1)

    def forward(self, x):  # x: (batch, seq_len, 1)
        x = self.input_proj(x)  # -> (batch, seq_len, input_dim)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = x.permute(1, 0, 2)  # -> (seq_len, batch, input_dim)
        x = self.transformer(x)
        x = x.permute(1, 0, 2).reshape(x.size(1), -1)  # -> (batch, seq_len * input_dim)
        return self.fc(x)
