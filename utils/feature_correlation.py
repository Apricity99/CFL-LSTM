import numpy as np
import pandas as pd
import pywt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt

models = {
    "Diff": [1, 7],
    "Holt-Winters": [(alpha, beta, gamma)
                     for alpha in [0.2, 0.4, 0.6, 0.8]
                     for beta in [0.2, 0.4, 0.6, 0.8]
                     for gamma in [0.2, 0.4, 0.6, 0.8]],
    "Historical Average": [7, 14, 21, 28],
    "Historical Median": [7, 14, 21, 28],
    "TSD": [7, 14, 21, 28],
    "TSD Median": [7, 14, 21, 28],
    "Wavelet": [1, 3, 5, 7]
}

def diff_forecast(data, days):
    return data.shift(days)

def holt_winters_forecast(data, params):
    alpha, beta, gamma = params
    try:
        seasonal_periods = 7
        if len(data) < 2 * seasonal_periods:
            return data.rolling(window=seasonal_periods).mean().fillna(method='bfill')
        model = ExponentialSmoothing(
            data,
            seasonal='add' if gamma > 0 else None,
            trend='add',
            seasonal_periods=seasonal_periods
        )
        fit = model.fit(
            smoothing_level=alpha,
            smoothing_trend=beta,
            smoothing_seasonal=gamma,
            optimized=False
        )
        return fit.fittedvalues.fillna(method='bfill')
    except Exception as e:
        return data.shift(1).fillna(method='bfill')

def historical_avg_forecast(data, window):
    return data.rolling(window=window, min_periods=1).mean().fillna(method='bfill')

def historical_median_forecast(data, window):
    return data.rolling(window=window, min_periods=1).median().fillna(method='bfill')

def tsd_forecast(data, window):
    data = pd.Series(data).dropna()
    if len(data) < 2 * window:
        return data.rolling(window=window, min_periods=1).mean()
    decomposition = seasonal_decompose(data, model='additive', period=window)
    trend = decomposition.trend.fillna(method='bfill')
    seasonal = decomposition.seasonal.fillna(method='bfill')
    return trend + seasonal

def tsd_median_forecast(data, window):
    data = pd.Series(data).dropna()
    if len(data) < 2 * window:
        trend = data.rolling(window=window, min_periods=1).median()
    else:
        trend = data.rolling(window=window, min_periods=1).median()
        detrended = data - trend
        seasonal = detrended.groupby(lambda x: x % window).median()
        seasonal_aligned = seasonal.reindex(data.index, method='ffill')
        return trend + seasonal_aligned
    return trend

def wavelet_forecast(data, window):
    level = int(np.log2(window)) if window > 1 else 3
    coeffs = pywt.wavedec(data, 'db1', level=level)
    coeffs_modified = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    reconstructed = pywt.waverec(coeffs_modified, 'db1')
    return pd.Series(reconstructed[:len(data)]).fillna(method='ffill').values

def align_series(data, forecast):
    data_series = pd.Series(data)
    forecast_series = pd.Series(forecast)
    forecast_series = forecast_series.reindex(data_series.index, method='ffill').fillna(method='bfill')
    return data_series.values, forecast_series.values

def f(alpha, beta, x):
    if x >= 0:
        return (np.exp(min(x, beta) * alpha) - 1) / (np.exp(beta * alpha) - 1)
    else:
        return -(np.exp(min(abs(x), beta) * alpha) - 1) / (np.exp(beta * alpha) - 1)

def amplify_features(feature, alpha, beta):
    return np.array([f(alpha, beta, x) for x in feature])

def extract_features(data):
    features = []
    model_feature_counts = {}
    data_series = pd.Series(data)
    for model_name, params_list in models.items():
        model_features = []
        for params in tqdm(params_list, desc=f"Extracting features for {model_name}", leave=False):
            if model_name == "Diff":
                forecast = diff_forecast(data_series, params)
            elif model_name == "Holt-Winters":
                forecast = holt_winters_forecast(data_series, params)
            elif model_name == "Historical Average":
                forecast = historical_avg_forecast(data_series, params)
            elif model_name == "Historical Median":
                forecast = historical_median_forecast(data_series, params)
            elif model_name == "TSD":
                forecast = tsd_forecast(data_series, params)
            elif model_name == "TSD Median":
                forecast = tsd_median_forecast(data_series, params)
            elif model_name == "Wavelet":
                forecast = wavelet_forecast(data_series, params)
            aligned_data, aligned_forecast = align_series(data_series, forecast)
            feature = aligned_data - aligned_forecast
            model_features.append(feature)
            features.append(feature)
        model_feature_counts[model_name] = len(model_features)
    print("各模型提取特征个数：", model_feature_counts)
    return features, model_feature_counts

def get_amplified_features_parallel(data, alpha=0.5, beta=10, num_workers=4):
    features, _ = extract_features(data)
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(amplify_features, [(feature, alpha, beta) for feature in features])
    return results

def attention_weights(features):
    scores = []
    for f in features:
        f_clean = f[~np.isnan(f)]
        if len(f_clean) == 0:
            scores.append(0)
        else:
            var = np.nanvar(f)
            kurt = stats.kurtosis(f, nan_policy='omit')
            if np.isnan(kurt):
                kurt = 0
            score = var + abs(kurt)
            # log变换抑制极端大值
            scores.append(np.log1p(score))
    scores = np.array(scores)
    # min-max归一化（可选）
    if np.max(scores) > np.min(scores):
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    total = np.nansum(scores)
    if total == 0 or np.isnan(total):
        return np.ones_like(scores) / len(scores)
    return scores / total

def select_top_features(features, weights, top_n=10):
    indices = np.argsort(weights)[-top_n:]
    return [features[i] for i in indices]

def FCC(af_x, af_y):
    l = len(af_x)
    R_G_G = np.sum(af_x ** 2)
    R_H_H = np.sum(af_y ** 2)
    if R_G_G == 0 or R_H_H == 0:
        return "无效", 0, 0
    norm = np.sqrt(R_G_G * R_H_H)
    cc_full = np.correlate(af_x, af_y, mode='full') / norm
    lags = np.arange(-l + 1, l)
    max_cc = np.max(cc_full)
    s2 = lags[np.argmax(cc_full)]
    min_cc = np.min(cc_full)
    s1 = lags[np.argmin(cc_full)]
    if abs(max_cc) >= abs(min_cc):
        correlation_type = "正相关" if max_cc > 0 else "负相关"
        return correlation_type, max_cc, s2
    else:
        correlation_type = "正相关" if min_cc > 0 else "负相关"
        return correlation_type, min_cc, s1

def FCC_single_computation(af_x, af_y):
    if len(af_x) == len(af_y):
        return FCC(af_x, af_y)
    else:
        return "无效", 0, 0

def FCC_parallel(af_xSet, af_ySet, coTHR, num_workers=4):
    args = [(af_x, af_y) for af_x in af_xSet for af_y in af_ySet]
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(FCC_single_computation, args)
    max_result = None
    max_ccV = 0
    for correlation_type, ccV, shiftV in results:
        if abs(ccV) > abs(max_ccV):
            max_ccV = ccV
            max_result = (correlation_type, ccV, shiftV)
    if max_result:
        correlation_type, ccV, shiftV = max_result
        if abs(ccV) >= coTHR:
            if shiftV == 0:
                return f"X ↔ Y ({correlation_type}, cc: {ccV:.2f}, shift: {shiftV})"
            elif shiftV < 0:
                return f"X → Y ({correlation_type}, cc: {ccV:.2f}, shift: {shiftV})"
            elif shiftV > 0:
                return f"Y → X ({correlation_type}, cc: {ccV:.2f}, shift: {shiftV})"
        else:
            return f"X ⟷ Y (无明显相关性, cc: {ccV:.2f}, shift: {shiftV})"
    else:
        return "没有找到相关性结果"

def parse_fcc_result(result_str):
    try:
        cc_index = result_str.find("cc:")
        shift_index = result_str.find("shift:")
        cc_str = result_str[cc_index + 4: result_str.find(",", cc_index)]
        shift_str = result_str[shift_index + 7: result_str.find(")", shift_index)]
        cc_val = float(cc_str.strip())
        tau_val = int(shift_str.strip())
    except Exception as e:
        cc_val = 0.0
        tau_val = 0
    return cc_val, tau_val

def compute_correlation(target_series, aux_series, coTHR=0.20, top_n=40):
    amplified_target = get_amplified_features_parallel(target_series)
    amplified_aux = get_amplified_features_parallel(aux_series)
    weights_target = attention_weights(amplified_target)
    weights_aux = attention_weights(amplified_aux)
    top_target = select_top_features(amplified_target, weights_target, top_n=top_n)
    top_aux = select_top_features(amplified_aux, weights_aux, top_n=top_n)
    result_str = FCC_parallel(top_target, top_aux, coTHR)
    cc, tau = parse_fcc_result(result_str)
    return cc, tau

def visualize_features(data, model_names=None, n_plot=1):
    """
    可视化特征提取结果。
    data: 原始序列（一维数组）
    model_names: 指定要可视化的模型名列表，默认全部
    n_plot: 可视化前n个特征（每种模型可能有多个特征）
    """
    features, model_feature_counts = extract_features(data)
    if model_names is None:
        model_names = list(model_feature_counts.keys())
    idx = 0
    plt.figure(figsize=(16, 2 * len(model_names)))
    for i, model in enumerate(model_names):
        count = model_feature_counts[model]
        for j in range(min(n_plot, count)):
            plt.subplot(len(model_names), n_plot, i * n_plot + j + 1)
            plt.plot(data, label='原始序列')
            plt.plot(features[idx], label=f'{model} 特征{j+1}')
            plt.title(f'{model} 特征{j+1}')
            plt.legend()
            plt.grid(True)
            idx += 1
        idx += (count - min(n_plot, count))  # 跳过未画的特征
    plt.tight_layout()
    plt.show() 