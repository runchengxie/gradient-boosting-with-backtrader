"""
工具函数模块
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split


def create_directory(directory_path: str):
    """创建目录（如果不存在）"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"创建目录: {directory_path}")


def print_data_info(data: pd.DataFrame, data_name: str = "数据"):
    """打印数据基本信息"""
    print(f"\n=== {data_name} 基本信息 ===")
    print(f"形状: {data.shape}")
    print(f"日期范围: {data.index.min()} 到 {data.index.max()}")
    
    if len(data) > 0:
        years_covered = (data.index.max() - data.index.min()).days / 365.25
        print(f"覆盖年数: {years_covered:.2f} 年")
    
    print(f"缺失值数量:")
    missing_counts = data.isnull().sum()
    missing_percentages = (missing_counts / len(data) * 100).round(2)
    
    for col in data.columns:
        if missing_counts[col] > 0:
            print(f"  {col}: {missing_counts[col]} ({missing_percentages[col]}%)")
    
    if missing_counts.sum() == 0:
        print("  无缺失值")


def print_feature_statistics(X: pd.DataFrame, feature_name: str = "特征"):
    """打印特征统计信息"""
    print(f"\n=== {feature_name} 统计信息 ===")
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    
    if X.shape[1] > 0:
        print("特征列表:")
        for i, col in enumerate(X.columns, 1):
            print(f"  {i:2d}. {col}")


def chronological_split(X: pd.DataFrame, y: pd.Series, 
                       test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """时间序列数据的按时间顺序分割"""
    # 确保数据按时间排序
    X_sorted = X.sort_index()
    y_sorted = y.loc[X_sorted.index]
    
    split_index = int(len(X_sorted) * (1 - test_size))
    
    X_train = X_sorted.iloc[:split_index].copy()
    X_test = X_sorted.iloc[split_index:].copy()
    y_train = y_sorted.iloc[:split_index].copy()
    y_test = y_sorted.iloc[split_index:].copy()
    
    print(f"\n=== 数据分割信息 ===")
    print(f"训练集: {X_train.shape[0]} 样本 ({X_train.index.min()} 到 {X_train.index.max()})")
    print(f"测试集: {X_test.shape[0]} 样本 ({X_test.index.min()} 到 {X_test.index.max()})")
    print(f"训练集目标分布: {y_train.value_counts(normalize=True).round(3).to_dict()}")
    print(f"测试集目标分布: {y_test.value_counts(normalize=True).round(3).to_dict()}")
    
    return X_train, X_test, y_train, y_test


def validate_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """验证数据质量"""
    quality_report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum(),
        'date_range': (data.index.min(), data.index.max()) if len(data) > 0 else (None, None),
        'issues': []
    }
    
    # 检查缺失值
    if quality_report['missing_values'] > 0:
        quality_report['issues'].append(f"发现 {quality_report['missing_values']} 个缺失值")
    
    # 检查重复行
    if quality_report['duplicate_rows'] > 0:
        quality_report['issues'].append(f"发现 {quality_report['duplicate_rows']} 个重复行")
    
    # 检查数值列的异常值
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in data.columns:
            # 检查无穷大值
            inf_count = np.isinf(data[col]).sum()
            if inf_count > 0:
                quality_report['issues'].append(f"列 '{col}' 包含 {inf_count} 个无穷大值")
            
            # 检查零值（对于应该为正数的列）
            if col in ['vol', 'amount', 'close', 'open', 'high', 'low']:
                zero_count = (data[col] == 0).sum()
                if zero_count > 0:
                    quality_report['issues'].append(f"列 '{col}' 包含 {zero_count} 个零值")
    
    return quality_report


def print_quality_report(quality_report: Dict[str, Any]):
    """打印数据质量报告"""
    print("\n=== 数据质量报告 ===")
    print(f"总行数: {quality_report['total_rows']}")
    print(f"总列数: {quality_report['total_columns']}")
    print(f"缺失值: {quality_report['missing_values']}")
    print(f"重复行: {quality_report['duplicate_rows']}")
    
    if quality_report['date_range'][0] is not None:
        print(f"日期范围: {quality_report['date_range'][0]} 到 {quality_report['date_range'][1]}")
    
    if quality_report['issues']:
        print("\n发现的问题:")
        for i, issue in enumerate(quality_report['issues'], 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✓ 未发现数据质量问题")


def calculate_basic_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """计算基本统计信息"""
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return {}
    
    stats = {
        'count': numeric_data.count(),
        'mean': numeric_data.mean(),
        'std': numeric_data.std(),
        'min': numeric_data.min(),
        'max': numeric_data.max(),
        'skewness': numeric_data.skew(),
        'kurtosis': numeric_data.kurtosis()
    }
    
    return stats


def print_basic_statistics(stats: Dict[str, Any]):
    """打印基本统计信息"""
    if not stats:
        print("没有数值型数据可计算统计信息")
        return
    
    print("\n=== 基本统计信息 ===")
    stats_df = pd.DataFrame(stats).round(4)
    print(stats_df)


def save_results_to_file(results: Dict[str, Any], filepath: str):
    """保存结果到文件"""
    try:
        import json
        
        # 转换numpy数组为列表，便于JSON序列化
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.int64, np.float64)):
                serializable_results[key] = float(value)
            elif isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict()
            elif isinstance(value, pd.Series):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {filepath}")
        
    except Exception as e:
        print(f"保存结果失败: {e}")


def load_results_from_file(filepath: str) -> Dict[str, Any]:
    """从文件加载结果"""
    try:
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"结果已从文件加载: {filepath}")
        return results
        
    except Exception as e:
        print(f"加载结果失败: {e}")
        return {}


def format_number(num: float, decimal_places: int = 2) -> str:
    """格式化数字显示"""
    if abs(num) >= 1e6:
        return f"{num/1e6:.{decimal_places}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{decimal_places}f}K"
    else:
        return f"{num:.{decimal_places}f}"


def calculate_returns_metrics(returns: pd.Series) -> Dict[str, float]:
    """计算收益率相关指标"""
    if returns.empty:
        return {}
    
    return {
        'total_return': (returns + 1).prod() - 1,
        'annualized_return': (returns + 1).prod() ** (252 / len(returns)) - 1,
        'volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
        'max_drawdown': calculate_max_drawdown(returns),
        'win_rate': (returns > 0).mean(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis()
    }


def calculate_max_drawdown(returns: pd.Series) -> float:
    """计算最大回撤"""
    if returns.empty:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()
