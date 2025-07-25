"""
CFL-LSTM: Cross-Feature Learning with LSTM for Time Series Forecasting

一个基于多序列特征融合的时间序列预测算法包
"""

__version__ = "1.0.0"
__author__ = "CFL-LSTM Team"
__email__ = "cfl.lstm@example.com"

from .src.CFL_LSTM_HX_alibaba import main as cfl_lstm_main

__all__ = ['cfl_lstm_main'] 