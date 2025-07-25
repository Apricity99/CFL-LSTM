import torch
import torch.nn as nn
from Baseline.GCNModel import GCNModel
from torch_geometric.data import Data
import inspect

HIDDEN_SIZE = 16
NUM_LAYERS = 2
DROPOUT = 0.3

def train_model(model, X_train, y_train, num_epochs=400, rho_weights=None):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # 判断模型 forward 是否支持 rho_weights
        sig = inspect.signature(model.forward)
        if 'rho_weights' in sig.parameters:
            outputs = model(X_train, rho_weights=rho_weights)
        else:
            outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def predict(model, X, scaler_target, device=None, rho_weights=None):
    model.eval()
    with torch.no_grad():
        if isinstance(model, GCNModel):
            edge_index = torch.tensor([
                [i, i + 1] for i in range(X.shape[0] - 1)
            ] + [
                [i + 1, i] for i in range(X.shape[0] - 1)
            ], dtype=torch.long).t().contiguous()
            node_features = X[:, 0, :]
            data = Data(x=node_features, edge_index=edge_index)
            if device is not None:
                data = data.to(device)
            preds = model(data).detach().cpu().numpy()
        else:
            sig = inspect.signature(model.forward)
            if 'rho_weights' in sig.parameters:
                preds = model(X, rho_weights=rho_weights).detach().cpu().numpy()
            else:
                preds = model(X).detach().cpu().numpy()
        return scaler_target.inverse_transform(preds)

def train_gcn_model(model, data, y, num_epochs=400):
    data = data
    y = y
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data).squeeze()
        loss = criterion(out, y.squeeze())
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"[GCN] Epoch {epoch+1}, Loss: {loss.item():.4f}")

def prepare_gcn_data(X, y):
    edge_index = torch.tensor([
        [i, i + 1] for i in range(X.shape[0] - 1)
    ] + [
        [i + 1, i] for i in range(X.shape[0] - 1)
    ], dtype=torch.long).t().contiguous()
    node_features = X.mean(dim=1)
    return Data(x=node_features, edge_index=edge_index), y 