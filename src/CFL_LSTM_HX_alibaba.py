import os
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
# 修改import路径以适应新的项目结构
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.CNNLSTMWeightedModel import CNNLSTMWeightedModel
from utils.data_utils import save_correlation_results, load_correlation_results, fill_missing, create_dataset_multisource, train_test_split
from utils.feature_correlation import compute_correlation
from utils.metrics import calculate_metrics
from utils.train_predict import train_model, predict
from utils.visualization import plot_cfl_lstm, save_predictions_to_csv
import matplotlib.pyplot as plt
import warnings
import random
from baseline.RNNModel import RNNModel
from baseline.SimpleLSTMModel import SimpleLSTMModel
from baseline.AttnLSTMModel import AttnLSTMModel
from baseline.DBiLSTMModel import DBiLSTMModel
from baseline.DBiGRUModel import DBiGRUModel
from baseline.GRUModel import GRUModel
from baseline.GCNModel import GCNModel
from baseline.TransformerModel import TransformerModel
from torch_geometric.data import Data
import multiprocessing as mp

# ===================== 超参数与配置 =====================
PARAMS = {
    'seed': 3,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'parent_directory': os.path.abspath(os.path.join(os.getcwd(), '.')),
    'dataset_directory': 'DataSet2',
    'corr_cache_filename': 'correlation_cache.csv',
    'coTHR': 0.20,
    'top_n': 40,
    'm': 3,  # 选取的辅助序列个数
    'time_step': 10,
    'target_filename': 'data.csv',
    'prediction_filename': 'cfl_lstm_predictions.csv',
    'font': 'STZhongsong',
}

# ===================== 工具函数 =====================
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_matplotlib(font):
    plt.rcParams['font.sans-serif'] = [font]
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings("ignore")

def get_dataset_directory(params):
    dataset_dir = params['dataset_directory']
    if os.path.isabs(dataset_dir):
        return dataset_dir
    else:
        return os.path.join(params['parent_directory'], dataset_dir)

def get_corr_cache_path(params):
    return os.path.join(get_dataset_directory(params), params['corr_cache_filename'])

def get_target_file_path(params):
    return os.path.join(get_dataset_directory(params), params['target_filename'])

def get_prediction_file_path(params):
    return os.path.join(get_dataset_directory(params), params['prediction_filename'])

# ===================== 主要流程函数 =====================
def correlation_analysis(params):
    """执行或读取相关性分析，返回辅助序列信息列表。"""
    corr_cache_path = get_corr_cache_path(params)
    dataset_directory = get_dataset_directory(params)
    if os.path.exists(corr_cache_path):
        print("检测到存在关联分析缓存文件，直接读取缓存...")
        auxiliary_info = load_correlation_results(corr_cache_path)
    else:
        print("没有检测到缓存文件，开始执行关联分析...")
        target_file = get_target_file_path(params)
        target_data = pd.read_csv(target_file, header=None).iloc[:10000]
        target_series = target_data[2].values[:10000]
        # 只读取DataSet2下以m开头的csv文件
        all_files = glob.glob(os.path.join(dataset_directory, 'm*.csv'))
        aux_files = [f for f in all_files if os.path.basename(f) != os.path.basename(target_file)]
        auxiliary_info = []
        print("开始对辅助序列进行关联分析：")
        for f in aux_files:
            aux_data = pd.read_csv(f, header=None).iloc[:10000]
            aux_series = aux_data[2].values[:10000]
            cc, tau = compute_correlation(target_series, aux_series, params['coTHR'], params['top_n'])
            direction = 1 if cc > 0 else -1 if cc < 0 else 0
            rho_val = abs(cc)
            auxiliary_info.append({
                'file': f,
                'rho': rho_val,
                'tau': tau,
                'direction': direction,
                'data': None
            })
            print(f"文件: {os.path.basename(f)}, rho: {rho_val:.2f}, "
                  f"tau: {tau}, direction: {direction}")
        save_correlation_results(auxiliary_info, corr_cache_path)
    return auxiliary_info

def select_and_align_auxiliary(auxiliary_info, params, target_series):
    """选择前m个辅助序列并对齐。返回对齐后的目标序列和辅助序列列表，以及实际m。"""
    rho_threshold = params.get('coTHR', 0.0)
    filtered_aux = [aux for aux in auxiliary_info if aux['rho'] > rho_threshold]
    filtered_aux = sorted(filtered_aux, key=lambda x: x['rho'], reverse=True)
    m = min(params['m'], len(filtered_aux))  # 自动调整m
    selected_aux = filtered_aux[:m]
    print(f"\n选取用于 LSTM 预测的辅助序列（rho > {rho_threshold}）：")
    for aux in selected_aux:
        print(f"文件: {os.path.basename(aux['file'])}, "
              f"rho: {aux['rho']:.2f}, tau: {aux['tau']}, direction: {aux['direction']}")
    for aux in selected_aux:
        if aux['data'] is None:
            aux_data = pd.read_csv(aux['file'], header=None).iloc[:10000]
            aux_series = aux_data[2].values[:10000]
            aux['data'] = aux_series.reshape(-1, 1)
    target_values = target_series.reshape(-1, 1)
    L = len(target_values)
    global_start, global_end = 0, L
    for aux in selected_aux:
        tau = aux['tau']
        start_i = max(0, -tau)
        end_i = min(L, L - tau)
        global_start = max(global_start, start_i)
        global_end = min(global_end, end_i)
    target_aligned = target_values[global_start:global_end]
    aligned_aux_list = []
    for aux in selected_aux:
        tau = aux['tau']
        direction = aux['direction']
        if direction == -1:
            print(f"检测到负相关: {os.path.basename(aux['file'])}，已做取反处理")
            aux['data'] = -aux['data']
        aligned_series = aux['data'][global_start + tau: global_end + tau]
        aligned_aux_list.append(aligned_series)
    return target_aligned, aligned_aux_list, selected_aux, m

def preprocess_and_create_dataset(target_aligned, aligned_aux_list, params):
    """归一化并创建多源数据集。"""
    target_aligned = fill_missing(target_aligned)
    aligned_aux_list = [fill_missing(s) for s in aligned_aux_list]
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(target_aligned)
    scalers_aux = []
    scaled_aux_list = []
    for series in aligned_aux_list:
        scaler_aux = MinMaxScaler(feature_range=(0, 1))
        scaled_series = scaler_aux.fit_transform(series)
        scalers_aux.append(scaler_aux)
        scaled_aux_list.append(scaled_series)
    X_with_aux, y_with_aux = create_dataset_multisource(scaled_target, scaled_aux_list, params['time_step'])
    X_with_aux = torch.FloatTensor(X_with_aux)
    y_with_aux = torch.FloatTensor(y_with_aux)
    return X_with_aux, y_with_aux, scaler_target

def train_and_evaluate(X_with_aux, y_with_aux, scaler_target, selected_aux, params, actual_m=None):
    """训练模型并评估，返回预测结果和指标。"""
    device = params['device']
    X_train, X_test, y_train, y_test = train_test_split(X_with_aux, y_with_aux)
    if actual_m is None:
        actual_m = params['m']
    input_size = 1 + actual_m
    model = CNNLSTMWeightedModel(input_size=input_size).to(device)
    rho_with_aux = torch.tensor([1.0] + [aux['rho'] for aux in selected_aux], dtype=torch.float32, device=device)
    print("\n训练有辅助（带偏移）权重门控模型...")
    train_model(model, X_train.to(device), y_train.to(device), rho_weights=rho_with_aux)
    train_preds = predict(model, X_train.to(device), scaler_target, device=device, rho_weights=rho_with_aux)
    test_preds = predict(model, X_test.to(device), scaler_target, device=device, rho_weights=rho_with_aux)
    y_train_actual = scaler_target.inverse_transform(y_train.numpy().reshape(-1, 1))
    y_test_actual = scaler_target.inverse_transform(y_test.numpy().reshape(-1, 1))
    columns = ["MSE", "RMSE", "MAE", "R2", "DA", "TU", "MAPE", "SMAPE", "MASE"]
    metrics_train = dict(zip(columns, calculate_metrics(y_train_actual, train_preds)))
    metrics_test = dict(zip(columns, calculate_metrics(y_test_actual, test_preds)))

    # === 多基线：只用目标序列，无辅助 ===
    print("\n训练多种基线（仅目标序列，无辅助）模型...")
    from utils.data_utils import create_dataset_single
    time_step = params['time_step']
    y_all = np.concatenate([y_train_actual, y_test_actual], axis=0)
    X_baseline, y_baseline = create_dataset_single(scaler_target.transform(y_all), time_step)
    X_baseline = torch.FloatTensor(X_baseline)
    y_baseline = torch.FloatTensor(y_baseline)
    train_size = len(y_train_actual) - time_step + 1
    X_train_b, X_test_b = X_baseline[:train_size], X_baseline[train_size:]
    y_train_b, y_test_b = y_baseline[:train_size], y_baseline[train_size:]

    BASELINE_MODELS = {
        "RNN": lambda: RNNModel(input_size=1, hidden_size=16, num_layers=1, dropout=0.5),
        "SimpleLSTM": lambda: SimpleLSTMModel(input_size=1, hidden_size=16, num_layers=1, dropout=0.5),
        "AttnLSTM": lambda: AttnLSTMModel(input_size=1, hidden_size=16, num_layers=2, dropout=0.5),
        "DBiLSTM": lambda: DBiLSTMModel(input_size=1, hidden_size=16, num_layers=1, dropout=0.5),
        "DBiGRU": lambda: DBiGRUModel(input_size=1, hidden_size=16, num_layers=1, dropout=0.5),
        "GRU": lambda: GRUModel(input_size=1, hidden_size=16, num_layers=1, dropout=0.5),
        "GCN": lambda: GCNModel(input_dim=1, hidden_dim=16),
        "Transformer": lambda: TransformerModel(input_dim=16, seq_len=time_step, nhead=4)
    }
    baseline_results = {}
    for name, model_fn in BASELINE_MODELS.items():
        print(f"训练基线模型: {name}")
        model = model_fn().to(device)
        if name == "GCN":
            # GCN特殊处理
            def prepare_gcn_data(X, y):
                edge_index = torch.tensor([
                    [i, i + 1] for i in range(X.shape[0] - 1)
                ] + [
                    [i + 1, i] for i in range(X.shape[0] - 1)
                ], dtype=torch.long).t().contiguous()
                node_features = X.mean(dim=1)
                return Data(x=node_features, edge_index=edge_index), y
            from torch_geometric.data import Data
            def train_gcn_model(model, data, y, num_epochs=400):
                data = data.to(device)
                y = y.to(device)
                model.train()
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                for epoch in range(num_epochs):
                    optimizer.zero_grad()
                    out = model(data).squeeze()
                    loss = criterion(out, y.squeeze())
                    loss.backward()
                    optimizer.step()
            gcn_data, gcn_y = prepare_gcn_data(X_train_b, y_train_b)
            gcn_data_test, gcn_y_test = prepare_gcn_data(X_test_b, y_test_b)
            train_gcn_model(model, gcn_data, gcn_y)
            gcn_data = gcn_data.to(device)
            gcn_data_test = gcn_data_test.to(device)
            train_preds_b = model(gcn_data).detach().cpu().numpy()
            test_preds_b = model(gcn_data_test).detach().cpu().numpy()
            y_train_actual_b = scaler_target.inverse_transform(y_train_b.numpy().reshape(-1, 1))
            y_test_actual_b = scaler_target.inverse_transform(y_test_b.numpy().reshape(-1, 1))
            train_preds_b = scaler_target.inverse_transform(train_preds_b.reshape(-1, 1))
            test_preds_b = scaler_target.inverse_transform(test_preds_b.reshape(-1, 1))
        elif name == "Transformer":
            train_model(model, X_train_b.to(device), y_train_b.to(device))
            train_preds_b = model(X_train_b.to(device)).detach().cpu().numpy()
            test_preds_b = model(X_test_b.to(device)).detach().cpu().numpy()
            y_train_actual_b = scaler_target.inverse_transform(y_train_b.numpy().reshape(-1, 1))
            y_test_actual_b = scaler_target.inverse_transform(y_test_b.numpy().reshape(-1, 1))
            train_preds_b = scaler_target.inverse_transform(train_preds_b.reshape(-1, 1))
            test_preds_b = scaler_target.inverse_transform(test_preds_b.reshape(-1, 1))
        else:
            train_model(model, X_train_b.to(device), y_train_b.to(device))
            train_preds_b = predict(model, X_train_b.to(device), scaler_target, device=device)
            test_preds_b = predict(model, X_test_b.to(device), scaler_target, device=device)
            y_train_actual_b = scaler_target.inverse_transform(y_train_b.numpy().reshape(-1, 1))
            y_test_actual_b = scaler_target.inverse_transform(y_test_b.numpy().reshape(-1, 1))
        metrics_train_b = dict(zip(columns, calculate_metrics(y_train_actual_b, train_preds_b)))
        metrics_test_b = dict(zip(columns, calculate_metrics(y_test_actual_b, test_preds_b)))
        baseline_results[name] = {"train": metrics_train_b, "test": metrics_test_b}

    return (train_preds, test_preds, y_train_actual, y_test_actual, metrics_train, metrics_test, baseline_results)

def save_and_plot_results(train_preds, test_preds, y_train_actual, y_test_actual, params,
                        baseline_results=None):
    """保存预测结果并绘图。"""
    columns = ["MSE", "RMSE", "MAE", "R2", "DA", "TU", "MAPE", "SMAPE", "MASE"]
    data_table = [
        calculate_metrics(y_train_actual, train_preds),
        calculate_metrics(y_test_actual, test_preds),
    ]
    index_labels = [
        "有辅助(带偏移)-Train-WEIGHTED",
        "有辅助(带偏移)-Test-WEIGHTED",
    ]
    # 如果有多基线，添加
    if baseline_results is not None:
        for name, res in baseline_results.items():
            data_table.append(list(res["train"].values()))
            data_table.append(list(res["test"].values()))
            index_labels.append(f"{name}-Train")
            index_labels.append(f"{name}-Test")
    df_results = pd.DataFrame(data_table, columns=columns, index=index_labels)
    print("\n==== CFL-LSTM权重门控模型误差指标（含多基线） ====")
    print(df_results.to_string(float_format="%.4f"))
    # 预测结果保存和绘图
    cfl_keys = [
        "有辅助(带偏移)-Train-WEIGHTED", "有辅助(带偏移)-Test-WEIGHTED"
    ]
    cfl_preds = [
        train_preds, test_preds
    ]
    min_len = min(map(len, [y_train_actual, y_test_actual] + cfl_preds))
    cfl_df = pd.DataFrame({
        key: preds[:min_len].flatten()
        for key, preds in zip(cfl_keys, cfl_preds)
    })
    cfl_df['True(Train)'] = y_train_actual.flatten()[:min_len]
    cfl_df['True(Test)']  = y_test_actual.flatten()[:min_len]
    save_predictions_to_csv(cfl_df, get_prediction_file_path(params))
    plot_cfl_lstm(cfl_df, cfl_keys, get_dataset_directory(params))

# ===================== 主函数 =====================
def main(params=None, return_metrics=False):
    if params is None:
        params = PARAMS.copy()
    set_seed(params['seed'])
    setup_matplotlib(params['font'])
    params['dataset_directory'] = get_dataset_directory(params)  # 绝对路径
    print("使用设备：", params['device'])
    auxiliary_info = correlation_analysis(params)
    target_file = get_target_file_path(params)
    target_data = pd.read_csv(target_file, header=None).iloc[:10000]
    target_series = target_data[2].values[:10000]
    target_aligned, aligned_aux_list, selected_aux, actual_m = select_and_align_auxiliary(auxiliary_info, params, target_series)
    X_with_aux, y_with_aux, scaler_target = preprocess_and_create_dataset(target_aligned, aligned_aux_list, params)
    (
        train_preds, test_preds, y_train_actual, y_test_actual, metrics_train, metrics_test, baseline_results
    ) = train_and_evaluate(
        X_with_aux, y_with_aux, scaler_target, selected_aux, params, actual_m)
    save_and_plot_results(
        train_preds, test_preds, y_train_actual, y_test_actual, params,
        baseline_results
    )
    if return_metrics:
        return {'train': metrics_train, 'test': metrics_test, 'baselines': baseline_results}

if __name__ == '__main__':
    main() 