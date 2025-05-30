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
        
        # 成交量分布
        plt.subplot(2, 2, 2)
        vol_col = 'vol' if 'vol' in data.columns else 'vol_log'
        if vol_col in data.columns:
            plt.hist(data[vol_col].dropna(), bins=50, density=True, alpha=0.7, color='green')
            plt.title('成交量分布')
            plt.xlabel('成交量')
            plt.ylabel('频率')
        else:
            plt.text(0.5, 0.5, '未找到成交量列', ha='center', va='center')
        
        # 收益率箱线图
        plt.subplot(2, 2, 3)
        if 'pct_change' in data.columns:
            plt.boxplot(data['pct_change'].dropna(), whis=1.5)
            plt.title('收益率箱线图')
            plt.ylabel('收益率 (%)')
        else:
            plt.text(0.5, 0.5, '未找到pct_change列', ha='center', va='center')
        
        # 成交量箱线图
        plt.subplot(2, 2, 4)
        if vol_col in data.columns:
            plt.boxplot(data[vol_col].dropna(), whis=1.5)
            plt.title('成交量箱线图')
            plt.ylabel('成交量')
        else:
            plt.text(0.5, 0.5, '未找到成交量列', ha='center', va='center')
        
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
    
    def plot_correlation_matrix(self, data: pd.DataFrame, method: str = 'pearson'):
        """绘制相关性矩阵热力图"""
        # 只选择数值型列
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            print("没有数值型数据可绘制相关性矩阵")
            return
        
        # 计算相关性矩阵
        if method == 'kendall':
            corr_matrix = numeric_data.corr(method='kendall')
        elif method == 'spearman':
            corr_matrix = numeric_data.corr(method='spearman')
        else:
            corr_matrix = numeric_data.corr(method='pearson')
        
        # 创建遮罩以隐藏上三角
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=self.figsize_large)
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                   annot=True, fmt='.2f', square=True, linewidths=.5,
                   annot_kws={"size": 8})
        plt.title(f'相关性矩阵热力图 ({method})')
        plt.tight_layout()
        self._show_and_save_plot("correlation_matrix.png")
    
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_scores: List[float],
                              title: str = "特征重要性",
                              top_n: int = 20):
        """绘制特征重要性图"""
        # 创建DataFrame并排序
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # 只显示前N个特征
        importance_df = importance_df.head(top_n)
        
        plt.figure(figsize=self.figsize_medium)
        plt.barh(range(len(importance_df)), importance_df['importance'].values.astype(float))
        plt.yticks(range(len(importance_df)), list(importance_df['feature'].values.astype(str)))
        plt.xlabel('重要性得分')
        plt.title(title)
        plt.gca().invert_yaxis()  # 最重要的特征在顶部
        plt.tight_layout()
        self._show_and_save_plot("feature_importance.png")
    
    def plot_target_distribution(self, target: pd.Series, title: str = "目标变量分布"):
        """绘制目标变量分布"""
        plt.figure(figsize=self.figsize_small)
        
        # 计算分布
        target_counts = target.value_counts()
        target_percentages = target.value_counts(normalize=True) * 100
        
        # 绘制柱状图
        bars = plt.bar(target_counts.index, target_counts.values.astype(float), 
                      color=['red', 'green'], alpha=0.7)
        
        # 添加数值标签
        for i, (count, percentage) in enumerate(zip(target_counts.values.astype(float), target_percentages.values.astype(float))):
            plt.text(i, count + max(target_counts) * 0.01, 
                    f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom')
        
        plt.xlabel('目标类别')
        plt.ylabel('样本数量')
        plt.title(title)
        plt.xticks([0, 1], ['负向移动 (0)', '正向移动 (1)'])
        plt.grid(True, axis='y', alpha=0.3)
        self._show_and_save_plot("target_distribution.png")
    
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
    
    def plot_learning_curves(self, cv_results: dict, model_name: str):
        """绘制学习曲线"""
        if not cv_results or 'mean_test_score' not in cv_results:
            print("没有交叉验证结果可绘制")
            return
        
        # 提取参数网格信息
        param_keys = [key for key in cv_results.keys() if key.startswith('param_')]
        if not param_keys:
            print("没有找到参数信息")
            return
        
        # 选择主要参数进行可视化
        main_param = param_keys[0]  # 使用第一个参数
        param_values = cv_results[main_param]
        test_scores = cv_results['mean_test_score']
        test_stds = cv_results['std_test_score']
        
        plt.figure(figsize=self.figsize_small)
        plt.errorbar(range(len(test_scores)), test_scores, yerr=test_stds,
                    marker='o', capsize=5, capthick=2)
        plt.xlabel('参数组合')
        plt.ylabel('交叉验证得分')
        plt.title(f'{model_name} 交叉验证结果')
        plt.grid(True, alpha=0.3)
        self._show_and_save_plot(f"learning_curves_{model_name}.png")
    
    def plot_residuals(self, y_true: pd.Series, y_pred_proba: np.ndarray):
        """绘制残差图（用于回归，这里适配为分类问题的误差分析）"""
        plt.figure(figsize=self.figsize_medium)
        
        # 计算预测误差
        errors = y_pred_proba - y_true
        
        # 残差散点图
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred_proba, errors, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('预测概率')
        plt.ylabel('预测误差')
        plt.title('预测误差散点图')
        plt.grid(True, alpha=0.3)
        
        # 误差分布直方图
        plt.subplot(1, 2, 2)
        plt.hist(errors, bins=30, alpha=0.7, color='orange')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.xlabel('预测误差')
        plt.ylabel('频率')
        plt.title('预测误差分布')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._show_and_save_plot("residuals.png")
