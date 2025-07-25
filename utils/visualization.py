import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_cfl_lstm(cfl_df, cfl_keys, dataset_directory):
    plt.figure(figsize=(12, 6))
    for key in cfl_keys:
        plt.plot(cfl_df[key], label=key)
    plt.plot(cfl_df['True(Test)'], '--', color='black', label='True(Test)')
    plt.title("CFL‑LSTM 各设置预测 vs. 真实值 (Test)")
    plt.xlabel("样本索引")
    plt.ylabel("目标值")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_directory, 'cfl_lstm_plot.png'))
    plt.show()

def plot_baseline(baseline_df, base_keys, dataset_directory):
    plt.figure(figsize=(14, 7))
    for key in base_keys:
        if key.endswith("-Test"):
            plt.plot(baseline_df[key], label=key)
    plt.plot(baseline_df['True'], '--', color='black', label='True(Test)')
    plt.title("基线模型(Test)预测 vs. 真实值")
    plt.xlabel("样本索引")
    plt.ylabel("目标值")
    plt.legend(loc='upper right', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_directory, 'baseline_plot.png'))
    plt.show()

def save_predictions_to_csv(df, path):
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"已保存预测到 {os.path.basename(path)}") 