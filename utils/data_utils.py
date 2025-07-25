import pandas as pd
import numpy as np

def save_correlation_results(auxiliary_info, save_path):
    rows = []
    for aux in auxiliary_info:
        rows.append({
            'file': aux['file'],
            'rho': aux['rho'],
            'tau': aux['tau'],
            'direction': aux['direction'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"关联分析结果已保存到: {save_path}")


def load_correlation_results(load_path):
    df = pd.read_csv(load_path, encoding='utf-8')
    aux_info_list = []
    for _, row in df.iterrows():
        aux_info_list.append({
            'file': row['file'],
            'rho': row['rho'],
            'tau': int(row['tau']),
            'direction': int(row['direction']),
            'data': None
        })
    return aux_info_list


def fill_missing(values):
    if np.any(np.isnan(values)):
        values = pd.Series(values.flatten()).interpolate().values.reshape(-1, 1)
    return values


def create_dataset_multisource(target, aux_list, time_step=10):
    X, y = [], []
    length = len(target)
    for i in range(length - time_step):
        target_window = target[i:i + time_step]
        aux_windows = [aux[i:i + time_step] for aux in aux_list]
        features = np.concatenate([target_window] + aux_windows, axis=1)
        X.append(features)
        y.append(target[i + time_step, 0])
    return np.array(X), np.array(y)


def create_dataset_single(target, time_step=10):
    X, y = [], []
    length = len(target)
    for i in range(length - time_step):
        window = target[i:i + time_step]
        X.append(window)
        y.append(target[i + time_step, 0])
    return np.array(X), np.array(y)


def train_test_split(X, y, train_ratio=0.8):
    train_size = int(len(X) * train_ratio)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:] 