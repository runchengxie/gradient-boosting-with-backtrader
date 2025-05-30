# 梯度提升股票预测项目 - 模块化版本

## 项目概述

本项目使用梯度提升算法预测股票价格走势，采用模块化设计，将原本的单一文件重构为多个专门的模块。

## 项目结构

```
gradient-boosting-with-backtrader/
│
├── main.py                    # 模块化主程序
├── main_old_backup.py        # 原始单文件版本备份（1154行）
├── environment.yml           # Conda环境配置
├── README.md                # 项目说明
├── results_summary.json     # 结果保存文件
│
├── src/                     # 源代码模块
│   ├── __init__.py         # 包初始化
│   ├── config.py           # 配置文件
│   ├── data_processing.py  # 数据处理模块
│   ├── feature_engineering.py  # 特征工程模块
│   ├── model_training.py   # 模型训练模块
│   ├── backtesting.py      # 回测模块
│   ├── visualization.py    # 可视化模块
│   └── utils.py            # 工具函数模块
│
├── downloaded_raw_data/     # 缓存数据目录
│   ├── *.parquet           # 缓存的数据文件
│   └── *.csv
│
└── catboost_info/          # CatBoost训练信息
    └── ...
```

## 模块说明

### 1. config.py - 配置管理
- 包含所有项目配置和常量
- 数据源配置、模型参数、可视化设置等
- 便于统一管理和修改参数

### 2. data_processing.py - 数据处理
- `DataCollector`: 数据获取类，支持缓存机制
- `DataProcessor`: 数据清洗、转换、特征创建
- 包含数据验证和完整性检查

### 3. feature_engineering.py - 特征工程
- `FeatureEngineer`: 特征工程类
- 聚类特征、特征选择、多重共线性检查
- 完整的特征选择流水线

### 4. model_training.py - 模型训练
- `ModelTrainer`: 模型训练类
- `ModelEvaluator`: 模型评估类
- 支持多种梯度提升算法（XGBoost、LightGBM、CatBoost等）

### 5. backtesting.py - 回测引擎
- `BacktestEngine`: 回测引擎类
- 支持多策略比较
- 性能指标计算和可视化

### 6. visualization.py - 可视化
- `DataVisualizer`: 数据可视化类
- 数据探索、模型结果、回测结果可视化
- 支持中文显示

### 7. utils.py - 工具函数
- 通用工具函数
- 数据质量检查、统计计算、文件操作等

## 优势对比

### 原始版本问题
- **单文件过大**: 1154行代码全部在一个文件中
- **缺乏模块化**: 功能混杂，难以维护
- **代码重复**: 缺乏抽象和复用
- **测试困难**: 无法进行单元测试
- **协作困难**: 团队开发容易冲突

### 模块化版本优势
- **代码组织清晰**: 按功能模块分离
- **易于维护**: 每个模块职责单一
- **可复用性强**: 模块可以独立使用
- **易于测试**: 支持单元测试
- **扩展性好**: 容易添加新功能
- **团队友好**: 不同成员可以负责不同模块

## 使用方法

### 1. 安装依赖
```bash
conda env create -f environment.yml
conda activate gradient-boosting
```

### 2. 设置API密钥
```bash
export TUSHARE_API_KEY="your_api_key_here"
```

### 3. 运行主程序
```bash
python main.py
```

### 4. 单独使用某个模块
```python
from src.data_processing import DataCollector
from src.visualization import DataVisualizer

# 数据收集
collector = DataCollector()
data = collector.fetch_data('000300.SH', '2020-01-01', '2023-12-31')

# 数据可视化
visualizer = DataVisualizer()
visualizer.plot_time_series(data)
```

## 配置说明

主要配置项在 `src/config.py` 中：

```python
# 数据配置
TICKER = '000300.SH'  # 股票代码
START_DATE = '2004-12-31'  # 开始日期
END_DATE = '2025-05-20'    # 结束日期

# 模型配置
TEST_SIZE = 0.2  # 测试集比例
POSITIVE_MOVE_THRESHOLD = 0.2  # 正向移动阈值

# 回测配置
INITIAL_CAPITAL = 100000  # 初始资金
```

## 扩展建议

1. **添加更多数据源**: 在 `DataCollector` 中添加其他数据源
2. **新增模型算法**: 在 `ModelTrainer` 中添加新的机器学习算法
3. **优化特征工程**: 在 `FeatureEngineer` 中添加更多特征创建方法
4. **完善回测策略**: 在 `BacktestEngine` 中添加更复杂的交易策略
5. **添加风险管理**: 加入止损、仓位管理等功能

## 注意事项

1. 确保已设置正确的Tushare API密钥
2. 首次运行会下载数据并缓存
3. 可以通过修改 `config.py` 来调整参数
4. 建议先在小数据集上测试

## 贡献指南

1. 每个模块都有明确的职责边界
2. 新功能应该添加到对应的模块中
3. 保持代码风格一致
4. 添加适当的文档和注释
5. 考虑向后兼容性
