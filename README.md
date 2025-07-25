# CFL-LSTM: Cross-Feature Learning with LSTM for Time Series Forecasting

## 📖 Algorithm Overview

CFL-LSTM (Cross-Feature Learning with LSTM) is a novel time series forecasting algorithm based on multi-sequence feature fusion. The algorithm achieves high-precision time series prediction through the following core steps:

### 🔍 Core Algorithm Workflow

1. **Feature Correlation Analysis**
   - Calculate correlation coefficients (ρ) between target sequence and all auxiliary sequences
   - Identify optimal time lag (τ) to capture temporal delay relationships
   - Select top-m most correlated auxiliary sequences based on correlation strength

2. **Sequence Alignment & Preprocessing**
   - Align target and auxiliary sequences according to time lags
   - Handle negatively correlated sequences (inversion operation)
   - Data normalization and missing value imputation

3. **Weighted Gating Fusion**
   - Utilize CNN-LSTM weighted model for feature fusion
   - Construct weight vectors based on correlation coefficients ρ
   - Dynamically weight and fuse multi-source time series features

4. **Prediction & Evaluation**
   - Train fusion model for time series forecasting
   - Compare with multiple baseline methods
   - Calculate comprehensive evaluation metrics (MSE, RMSE, MAE, R², DA, TU, MAPE, SMAPE, MASE)

## 📁 Project Structure

```
CFL-LSTM-GitHub/
├── main.py                       # Main execution file
├── README.md                     # Project documentation (Chinese)
├── README_EN.md                  # Project documentation (English)
├── requirements.txt              # Dependencies list
├── LICENSE                       # MIT License
├── __init__.py                  # Python package initialization
├── src/                         # Core algorithm implementation
│   ├── CFL_LSTM_HX_alibaba.py   # 🎯 Main algorithm file
│   ├── CNNLSTMWeightedModel.py   # Weighted gating model
│   └── __init__.py
├── baseline/                    # Baseline methods (excluding CNN-LSTM)
│   ├── SimpleLSTMModel.py        # Simple LSTM
│   ├── RNNModel.py              # Standard RNN
│   ├── GRUModel.py              # GRU model
│   ├── AttnLSTMModel.py         # Attention LSTM
│   ├── DBiLSTMModel.py          # Bidirectional LSTM
│   ├── DBiGRUModel.py           # Bidirectional GRU
│   ├── GCNModel.py              # Graph Convolutional Network
│   ├── TransformerModel.py       # Transformer
│   └── __init__.py
├── utils/                       # Utility functions
│   ├── data_utils.py            # Data processing utilities
│   ├── feature_correlation.py   # Feature correlation analysis
│   ├── metrics.py               # Evaluation metrics
│   ├── train_predict.py         # Training and prediction
│   ├── visualization.py         # Visualization tools
│   └── __init__.py
├── data/                        # Dataset
│   ├── data.csv                 # Target time series
│   └── m_*.csv                  # Auxiliary time series files
├── docs/                        # Documentation directory
└── README.md                    # Project documentation
```

## 🚀 Quick Start

### System Requirements

- Python 3.7+
- PyTorch 1.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- torch-geometric (for GCN baseline)

### Installation

```bash
pip install -r requirements.txt
```

### Running the Algorithm

```python
# Import main algorithm
from src.CFL_LSTM_HX_alibaba import main

# Run with default parameters
main()

# Run with custom parameters
params = {
    'seed': 3,
    'dataset_directory': 'data',
    'coTHR': 0.20,        # Correlation threshold
    'top_n': 40,          # Correlation analysis window size
    'm': 3,               # Number of auxiliary sequences to select
    'time_step': 10,      # LSTM time step
}
main(params)
```

## 📊 Algorithm Features

### 🔬 Key Innovations

1. **Multi-sequence Feature Fusion**: Automatically discover and utilize information from multiple auxiliary sequences
2. **Time Lag Alignment**: Identify and handle temporal delay relationships between sequences
3. **Weighted Gating Mechanism**: Dynamically weight and fuse features based on correlation strength
4. **Negative Correlation Processing**: Intelligently handle information contribution from negatively correlated sequences

### 📈 Performance Advantages

- **High Accuracy**: Significantly improves prediction accuracy compared to single-sequence methods
- **Robustness**: Strong resistance to noise and outliers
- **Interpretability**: Correlation analysis provides clear feature importance explanation
- **Generalizability**: Applicable to various time series forecasting tasks

## 🧪 Experimental Results

The algorithm is compared with the following baseline methods across multiple datasets:

### Baseline Methods
- **SimpleLSTM**: Standard LSTM network
- **RNN**: Recurrent Neural Network
- **GRU**: Gated Recurrent Unit
- **AttnLSTM**: LSTM with attention mechanism
- **DBiLSTM**: Bidirectional LSTM
- **DBiGRU**: Bidirectional GRU
- **GCN**: Graph Convolutional Network
- **Transformer**: Self-attention mechanism model

### Evaluation Metrics
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **DA**: Directional Accuracy
- **TU**: Turning Point Accuracy
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error

## 📝 Usage Instructions

### Data Format

Data files should be in CSV format with the following structure:
- Target sequence file: `data.csv`, column 3 contains target time series values
- Auxiliary sequence files: `m_*.csv`, column 3 contains auxiliary time series values

### Parameter Configuration

Main parameters:
- `coTHR`: Correlation threshold for filtering effective auxiliary sequences
- `m`: Number of auxiliary sequences to select
- `time_step`: LSTM time step length
- `top_n`: Window size for correlation analysis

### Output Results

After running the algorithm, the following will be generated:
- Prediction results CSV file
- Visualization charts
- Performance evaluation report
- Correlation analysis cache file

## 🎯 Algorithm Workflow

### Phase 1: Correlation Analysis
```python
# Calculate correlation between target and auxiliary sequences
for aux_file in auxiliary_files:
    correlation_coeff, time_lag = compute_correlation(target, aux_sequence)
    if correlation_coeff > threshold:
        selected_auxiliaries.append(aux_file)
```

### Phase 2: Sequence Alignment
```python
# Align sequences based on time lags
for aux in selected_auxiliaries:
    aligned_aux = align_sequence(aux, time_lag)
    if correlation < 0:
        aligned_aux = -aligned_aux  # Handle negative correlation
```

### Phase 3: Weighted Fusion
```python
# Create weight vector based on correlation coefficients
rho_weights = [1.0] + [aux.rho for aux in selected_auxiliaries]
model = CNNLSTMWeightedModel(input_size=1+m)
predictions = model(features, rho_weights=rho_weights)
```

### Phase 4: Model Training & Evaluation
```python
# Train model and evaluate against baselines
model.train()
for baseline in baseline_models:
    baseline_results = evaluate_model(baseline, test_data)
compare_results(cfl_lstm_results, baseline_results)
```

## 📚 Citation

If you use this algorithm in your research, please cite:

```bibtex
@article{cfl_lstm_2024,
  title={CFL-LSTM: Cross-Feature Learning with LSTM for Time Series Forecasting},
  author={CFL-LSTM Team},
  journal={Your Conference/Journal},
  year={2024},
  pages={1--12}
}
```

## 🔧 Advanced Usage

### Custom Model Configuration

```python
# Configure custom parameters
custom_params = {
    'seed': 42,
    'coTHR': 0.15,           # Lower threshold for more auxiliary sequences
    'm': 5,                  # Select top-5 sequences
    'time_step': 15,         # Longer time window
    'hidden_size': 32,       # Larger hidden dimension
    'num_layers': 3,         # Deeper network
    'dropout': 0.2,          # Lower dropout rate
}

# Run with custom configuration
results = main(custom_params, return_metrics=True)
```

### Batch Processing Multiple Datasets

```python
import os
import glob

# Process multiple datasets
dataset_dirs = glob.glob('datasets/*/data')
results_summary = {}

for dataset_dir in dataset_dirs:
    dataset_name = os.path.basename(os.path.dirname(dataset_dir))
    params = PARAMS.copy()
    params['dataset_directory'] = dataset_dir
    
    results = main(params, return_metrics=True)
    results_summary[dataset_name] = results
    
# Generate comparison report
generate_comparison_report(results_summary)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA Out of Memory**: Reduce batch size or use CPU
   ```python
   params['device'] = torch.device('cpu')
   ```

3. **Data Format Error**: Check CSV file structure
   - Ensure column 3 contains numerical time series data
   - Remove headers if present

4. **Low Correlation Warning**: Increase auxiliary sequences
   ```python
   params['coTHR'] = 0.10  # Lower threshold
   params['m'] = 10        # More sequences
   ```

## 📈 Performance Tips

1. **Data Preprocessing**: Clean and normalize data before training
2. **Feature Selection**: Use domain knowledge to select relevant auxiliary sequences
3. **Hyperparameter Tuning**: Experiment with different correlation thresholds and model architectures
4. **Cross-Validation**: Use multiple train-test splits for robust evaluation

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/CFL-LSTM.git
cd CFL-LSTM

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/ baseline/ utils/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Email**: cfl.lstm@example.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/CFL-LSTM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/CFL-LSTM/discussions)

## 🙏 Acknowledgments

- Thanks to all contributors who helped improve this project
- Special thanks to the open-source community for providing excellent baseline implementations
- Inspired by recent advances in multi-modal time series forecasting research

---

**Star ⭐ this repository if you find it helpful!** 
