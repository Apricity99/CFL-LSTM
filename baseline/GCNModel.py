# Baseline/STGCNModel.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=1):
        super(GCNModel, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.gcn1(x, edge_index))
        x = self.relu(self.gcn2(x, edge_index))
        return self.fc(x)
