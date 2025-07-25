import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def direction_accuracy(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    n = len(y_true)
    if n < 2:
        return 0.0
    correct = 0
    for t in range(1, n):
        true_diff = y_true[t] - y_true[t - 1]
        pred_diff = y_pred[t] - y_pred[t - 1]
        if true_diff * pred_diff > 0:
            correct += 1
    return correct / (n - 1)

def theils_u_stat(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    n = len(y_true)
    eps = 1e-10
    numerator = 0.0
    for t in range(n):
        if abs(y_true[t]) > eps:
            numerator += ((y_pred[t] - y_true[t]) / (y_true[t])) ** 2
    denominator = 0.0
    for t in range(n - 1):
        if abs(y_true[t]) > eps:
            denominator += ((y_true[t + 1] - y_true[t]) / (y_true[t])) ** 2
    if denominator < eps:
        return np.nan
    U = np.sqrt((numerator / n)) / np.sqrt((denominator / (n - 1)))
    return U

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    eps = 1e-10
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    eps = 1e-10
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > eps
    if not np.any(mask):
        return np.nan
    smape = 2.0 * np.mean(numerator[mask] / denominator[mask]) * 100
    return smape

def mean_absolute_scaled_error(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    n = len(y_true)
    mae_model = np.mean(np.abs(y_pred - y_true))
    if n < 2:
        return np.nan
    mae_naive = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    if mae_naive < 1e-10:
        return np.nan
    return mae_model / mae_naive

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    da = direction_accuracy(y_true, y_pred)
    tu_val = theils_u_stat(y_true, y_pred)
    mape_val = mean_absolute_percentage_error(y_true, y_pred)
    smape_val = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    mase_val = mean_absolute_scaled_error(y_true, y_pred)
    return mse, rmse, mae, r2, da, tu_val, mape_val, smape_val, mase_val 