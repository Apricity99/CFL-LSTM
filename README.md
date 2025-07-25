# CFL-LSTM: Cross-Feature Learning with LSTM for Time Series Forecasting

## 📖 算法简介

CFL-LSTM (Cross-Feature Learning with LSTM) 是一种基于多序列特征融合的时间序列预测算法。该算法通过以下核心步骤实现高精度的时间序列预测：

### 🔍 核心算法流程

1. **特征关联分析 (Feature Correlation Analysis)**
   - 计算目标序列与所有辅助序列的相关性系数 (ρ)
   - 识别最优时间偏移量 (τ) 以捕获时间延迟关系
   - 根据相关性强度选择前 m 个最相关的辅助序列

2. **序列对齐与预处理 (Sequence Alignment & Preprocessing)**
   - 根据时间偏移量对齐目标序列和辅助序列
   - 处理负相关序列（取反操作）
   - 数据归一化和缺失值填补

3. **权重门控融合 (Weighted Gating Fusion)**
   - 使用 CNN-LSTM 加权模型进行特征融合
   - 根据相关性系数 ρ 构造权重向量
   - 动态加权融合多源时间序列特征

4. **预测与评估 (Prediction & Evaluation)**
   - 训练融合模型进行时间序列预测
   - 与多种基线方法进行对比评估
   - 计算多项评估指标 (MSE, RMSE, MAE, R², DA, TU, MAPE, SMAPE, MASE)

## 📁 项目结构

```
CFL-LSTM-GitHub/
├── src/                          # 核心算法实现
│   ├── CFL_LSTM_HX_alibaba.py    # 主算法文件
│   └── CNNLSTMWeightedModel.py    # 权重门控模型
├── baseline/                     # 基线方法实现
│   ├── SimpleLSTMModel.py         # 简单LSTM
│   ├── RNNModel.py               # 标准RNN
│   ├── GRUModel.py               # GRU模型
│   ├── AttnLSTMModel.py          # 注意力LSTM
│   ├── DBiLSTMModel.py           # 双向LSTM
│   ├── DBiGRUModel.py            # 双向GRU
│   ├── GCNModel.py               # 图卷积网络
│   └── TransformerModel.py        # Transformer
├── utils/                        # 工具函数
│   ├── data_utils.py             # 数据处理工具
│   ├── feature_correlation.py    # 特征关联分析
│   ├── metrics.py                # 评估指标计算
│   ├── train_predict.py          # 训练预测函数
│   └── visualization.py          # 可视化工具
├── data/                         # 数据集
│   ├── data.csv                  # 目标时间序列
│   └── m_*.csv                   # 辅助时间序列文件
├── docs/                         # 文档目录
└── README.md                     # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- torch-geometric (用于GCN基线)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行算法

```python
# 导入主算法
from src.CFL_LSTM_HX_alibaba import main

# 使用默认参数运行
main()

# 自定义参数运行
params = {
    'seed': 3,
    'dataset_directory': 'data',
    'coTHR': 0.20,        # 相关性阈值
    'top_n': 40,          # 关联分析窗口大小
    'm': 3,               # 选择的辅助序列数量
    'time_step': 10,      # 时间步长
}
main(params)
```

## 📊 算法特点

### 🔬 创新点

1. **多序列特征融合**: 自动发现并利用多个辅助序列的信息
2. **时间偏移对齐**: 识别并处理序列间的时间延迟关系
3. **权重门控机制**: 根据相关性强度动态加权融合特征
4. **负相关处理**: 智能处理负相关序列的信息贡献

### 📈 性能优势

- **高精度**: 相比单序列方法显著提升预测精度
- **鲁棒性**: 对噪声和异常值具有较强的抗干扰能力
- **可解释性**: 相关性分析提供清晰的特征重要性解释
- **通用性**: 适用于多种时间序列预测任务

## 🧪 实验结果

算法在多个数据集上与以下基线方法进行对比：

### 基线方法
- **SimpleLSTM**: 标准LSTM网络
- **RNN**: 循环神经网络
- **GRU**: 门控循环单元
- **AttnLSTM**: 带注意力机制的LSTM
- **DBiLSTM**: 双向LSTM
- **DBiGRU**: 双向GRU
- **GCN**: 图卷积网络
- **Transformer**: 自注意力机制模型

### 评估指标
- **MSE**: 均方误差
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **R²**: 决定系数
- **DA**: 方向准确性
- **TU**: 转折点准确性
- **MAPE**: 平均绝对百分比误差
- **SMAPE**: 对称平均绝对百分比误差
- **MASE**: 平均绝对标度误差

## 📝 使用说明

### 数据格式

数据文件应为CSV格式，包含以下结构：
- 目标序列文件: `data.csv`，第3列为目标时间序列值
- 辅助序列文件: `m_*.csv`，第3列为辅助时间序列值

### 参数配置

主要参数说明：
- `coTHR`: 相关性阈值，用于筛选有效辅助序列
- `m`: 选择的辅助序列数量
- `time_step`: LSTM时间步长
- `top_n`: 关联分析时的窗口大小

### 输出结果

算法运行后会生成：
- 预测结果CSV文件
- 可视化图表
- 性能评估报告
- 关联分析缓存文件

## 📚 相关论文

如果使用本算法，请引用相关论文：

```bibtex
@article{cfl_lstm,
  title={CFL-LSTM: Cross-Feature Learning with LSTM for Time Series Forecasting},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024}
}
```

## 📄 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📞 联系方式

如有问题，请联系：[your-email@example.com] 