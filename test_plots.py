#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试所有绘图功能是否正确保存到plots文件夹
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from src.visualization import DataVisualizer
from src.backtesting import BacktestEngine
from src.model_training import ModelTrainer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def test_visualization():
    """测试可视化模块"""
    print("测试可视化模块...")
    
    # 创建测试数据
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'pct_change': np.random.randn(100) * 0.02,
        'vol': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # 测试可视化
    viz = DataVisualizer()
    viz.plot_data_overview(data)
    viz.plot_time_series(data)
    
    # 测试预测分布图
    y_true = pd.Series(np.random.choice([0, 1], 100))
    y_pred_proba = np.random.random(100)
    viz.plot_prediction_distribution(y_true, y_pred_proba)
    
    print("可视化模块测试完成")

def test_backtesting():
    """测试回测模块"""
    print("测试回测模块...")
    
    # 创建测试数据
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'close': np.random.randn(50).cumsum() + 100
    }, index=dates)
    
    # 测试回测
    engine = BacktestEngine()
    engine.run_buy_and_hold_backtest(data)
    
    # 添加一个简单的策略测试
    signals = np.random.choice([0, 1], 50)
    engine.run_strategy_backtest(data, "测试策略", signals)
    
    # 测试绘图
    engine.plot_portfolio_values()
    engine.plot_drawdown("测试策略")
    
    print("回测模块测试完成")

def test_model_training():
    """测试模型训练模块"""
    print("测试模型训练模块...")
    
    # 创建测试数据
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 转换为DataFrame
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    
    # 测试模型训练
    trainer = ModelTrainer(random_state=42)
    
    # 只训练一个简单的模型来测试
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算指标
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    
    # 创建评估结果
    evaluation_results = {
        '测试模型': {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
    }
    
    # 测试绘图
    trainer.plot_roc_curves(evaluation_results)
    trainer._plot_confusion_matrix(cm, '测试模型')
    
    print("模型训练模块测试完成")

def main():
    """主测试函数"""
    print("开始测试所有绘图功能...")
    print("=" * 50)
    
    # 确保plots目录存在
    os.makedirs('plots', exist_ok=True)
    
    try:
        test_visualization()
        print()
        test_backtesting()
        print()
        test_model_training()
        
        print("=" * 50)
        print("所有测试完成！")
        
        # 列出plots文件夹中的所有文件
        print("\nplots文件夹中的文件:")
        for file in os.listdir('plots'):
            if file.endswith('.png'):
                print(f"  - {file}")
                
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()