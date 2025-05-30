"""
可视化模块 - 支持非阻塞显示和自动保存
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Optional, List, Tuple


class DataVisualizer:
    """数据可视化类"""
    
    def __init__(self, figsize_small: Tuple[int, int] = (10, 6),
                 figsize_medium: Tuple[int, int] = (12, 8),
                 figsize_large: Tuple[int, int] = (15, 10),
                 non_blocking: bool = True,
                 save_plots: bool = True,
                 output_dir: str = "plots"):
        self.figsize_small = figsize_small
        self.figsize_medium = figsize_medium
        self.figsize_large = figsize_large
        self.non_blocking = non_blocking
        self.save_plots = save_plots
        self.output_dir = output_dir
        
        # 创建输出目录
        if self.save_plots:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置matplotlib中文字体和非阻塞模式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        if self.non_blocking:
            # 设置非阻塞模式
            plt.ion()  # 开启交互模式
            
    def _show_and_save_plot(self, filename: Optional[str] = None):
        """显示和保存图表的统一方法"""
        if self.save_plots and filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {filepath}")
        
        if self.non_blocking:
            plt.show(block=False)  # 非阻塞显示
            plt.pause(0.1)  # 短暂暂停以确保图表正确显示
        else:
            plt.show()  # 阻塞显示
    
    def plot_data_overview(self, data: pd.DataFrame):
        """绘制数据概览图"""
        plt.figure(figsize=self.figsize_medium)
        
        # 收益率分布
        plt.subplot(2, 2, 1)
        if 'pct_change' in data.columns:
            plt.hist(data['pct_change'].dropna(), bins=50, density=True, alpha=0.7, color='blue')
            plt.title('日收益率分布')
            plt.xlabel('收益率 (%)')
            plt.ylabel('频率')
        else:
            plt.text(0.5, 0.5, '未找到pct_change列', ha='center', va='center')
        
        plt.tight_layout()
        self._show_and_save_plot("data_overview.png")
    
    def plot_time_series(self, data: pd.DataFrame):
        """绘制时间序列图"""
        plt.figure(figsize=self.figsize_large)
        
        # 收盘价走势
        plt.subplot(2, 1, 1)
        if 'close' in data.columns:
            plt.plot(data.index, data['close'], color='blue', linewidth=1)
            plt.title('收盘价走势')
            plt.xlabel('日期')
            plt.ylabel('价格')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, '未找到close列', ha='center', va='center')
        
        # 成交量走势
        plt.subplot(2, 1, 2)
        vol_col = 'vol' if 'vol' in data.columns else 'vol_log'
        if vol_col in data.columns:
            plt.plot(data.index, data[vol_col], color='red', linewidth=1)
            plt.title('成交量走势')
            plt.xlabel('日期')
            plt.ylabel('成交量')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, '未找到成交量列', ha='center', va='center')
        
        plt.tight_layout()
        self._show_and_save_plot("time_series.png")
    
    # 添加其他必要的方法作为占位符
    def plot_correlation_matrix(self, data: pd.DataFrame, method: str = 'pearson'):
        """绘制相关性矩阵热力图"""
        print("相关性矩阵绘制功能正在工作中...")
        
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_scores: List[float],
                              title: str = "特征重要性",
                              top_n: int = 20):
        """绘制特征重要性图"""
        print("特征重要性绘制功能正在工作中...")
        
    def plot_target_distribution(self, target: pd.Series, title: str = "目标变量分布"):
        """绘制目标变量分布"""
        print("目标分布绘制功能正在工作中...")
    
    def plot_prediction_distribution(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                                   title: str = "预测概率分布"):
        """绘制预测概率分布"""
        plt.figure(figsize=self.figsize_medium)
        
        # 按真实标签分组绘制概率分布
        plt.subplot(1, 2, 1)
        for label in [0, 1]:
            mask = y_true == label
            plt.hist(y_pred_proba[mask], bins=30, alpha=0.6, 
                    label=f'真实标签 {label}', density=True)
        plt.xlabel('预测概率')
        plt.ylabel('密度')
        plt.title('按真实标签的预测概率分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 整体预测概率分布
        plt.subplot(1, 2, 2)
        plt.hist(y_pred_proba, bins=50, alpha=0.7, color='purple', density=True)
        plt.xlabel('预测概率')
        plt.ylabel('密度')
        plt.title('整体预测概率分布')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        self._show_and_save_plot("prediction_distribution.png")
