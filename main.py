#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CFL-LSTM 主运行文件
Cross-Feature Learning with LSTM for Time Series Forecasting

使用方法:
    python main.py

作者: CFL-LSTM团队
日期: 2024年
"""

import os
import sys

# 添加项目路径到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """主函数：运行CFL-LSTM算法"""
    
    print("=" * 60)
    print("CFL-LSTM: Cross-Feature Learning with LSTM")
    print("时间序列预测算法")
    print("=" * 60)
    
    try:
        # 导入CFL-LSTM主算法
        from src.CFL_LSTM_HX_alibaba import main as cfl_main
        
        # 配置参数
        params = {
            'seed': 3,                    # 随机种子
            'dataset_directory': 'data',  # 数据目录
            'coTHR': 0.20,               # 相关性阈值
            'top_n': 40,                 # 关联分析窗口大小
            'm': 3,                      # 选择的辅助序列数量
            'time_step': 10,             # LSTM时间步长
            'target_filename': 'data.csv',    # 目标序列文件名
            'prediction_filename': 'cfl_lstm_predictions.csv',  # 预测结果文件名
            'font': 'SimHei',            # 中文字体
        }
        
        print(f"算法参数配置:")
        print(f"  - 随机种子: {params['seed']}")
        print(f"  - 数据目录: {params['dataset_directory']}")
        print(f"  - 相关性阈值: {params['coTHR']}")
        print(f"  - 辅助序列数量: {params['m']}")
        print(f"  - 时间步长: {params['time_step']}")
        print("-" * 60)
        
        # 运行算法
        print("开始运行CFL-LSTM算法...")
        results = cfl_main(params, return_metrics=True)
        
        print("\n" + "=" * 60)
        print("算法运行完成!")
        print("结果已保存至data目录")
        print("=" * 60)
        
        if results:
            print("\n算法性能指标:")
            train_metrics = results['train']
            test_metrics = results['test']
            
            print(f"训练集 - MSE: {train_metrics['MSE']:.4f}, R²: {train_metrics['R2']:.4f}")
            print(f"测试集 - MSE: {test_metrics['MSE']:.4f}, R²: {test_metrics['R2']:.4f}")
            
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所有依赖包: pip install -r requirements.txt")
        return 1
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保data目录包含所需的数据文件")
        return 1
        
    except Exception as e:
        print(f"运行时错误: {e}")
        print("请检查数据格式和配置参数")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 