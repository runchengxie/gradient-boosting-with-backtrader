# 基于梯度提升算法的股票价格预测与交易策略回测

一个使用梯度提升机器学习算法预测股票价格并进行交易策略回测的完整项目。该项目基于中证300指数(CSI 300, 000300.SH)数据，通过技术指标分析和特征工程来预测股票的短期走势。

## 📈 项目概述

本项目实现了一个完整的机器学习交易系统，包含以下核心功能：

- **数据获取**: 使用Tushare API获取中证300指数的历史数据和技术指标
- **特征工程**: 创建交互特征、数学变换和聚类特征
- **模型训练**: 使用多种梯度提升算法(XGBoost, LightGBM, CatBoost)
- **特征选择**: 采用漏斗式特征选择方法(VIF、统计测试、RFE)
- **策略回测**: 基于模型预测结果进行交易策略回测

## 🏗️ 项目架构

项目采用7步模型构建流程：

1. **问题理解** - 定义预测目标和评估指标
2. **数据收集和准备** - 获取股票数据和技术指标
3. **数据探索和可视化** - 分析数据分布和相关性
4. **数据清洗** - 处理缺失值和异常值
5. **数据变换和特征工程** - 创建预测特征
6. **模型训练和评估** - 训练多个机器学习模型
7. **交易策略回测** - 验证模型在实际交易中的表现

## 🚀 快速开始

### 环境要求

- Python 3.9+
- Conda或Miniconda

### 安装依赖

1. 克隆项目到本地：
```bash
git clone https://github.com/yourusername/gradient-boosting-with-backtrader.git
cd gradient-boosting-with-backtrader
```

2. 创建并激活Conda环境：
```bash
conda env create -f environment.yml
conda activate exam-3-env
```

### 配置API密钥

在运行代码之前，需要设置Tushare API密钥：

```bash
# Windows PowerShell
$env:TUSHARE_API_KEY="your_tushare_api_key_here"

# Linux/Mac
export TUSHARE_API_KEY="your_tushare_api_key_here"
```

> 💡 **获取Tushare API密钥**: 访问 [Tushare官网](https://tushare.pro/) 注册账户并获取免费API密钥

### 运行项目

```bash
python main.py
```

## 📊 核心功能

### 数据处理
- **数据源**: 中证300指数(000300.SH)历史数据
- **时间范围**: 2004年12月31日 - 2025年5月20日
- **技术指标**: 包含30+个技术指标(MA, RSI, MACD, Bollinger Bands等)
- **数据缓存**: 自动缓存下载的数据，避免重复API调用

### 特征工程
- **数学变换**: 对成交量和成交额进行对数变换
- **交互特征**: 创建趋势与动量、波动率的交互特征
- **聚类特征**: 使用K-Means聚类添加市场状态特征
- **特征选择**: VIF过滤 → 统计测试 → 递归特征消除

### 机器学习模型
- **XGBoost**: 极端梯度提升算法
- **LightGBM**: 轻量级梯度提升算法
- **CatBoost**: 处理类别特征的梯度提升算法
- **网格搜索**: 自动调优模型超参数

### 回测系统
- **预测信号**: 基于模型输出生成买卖信号
- **策略对比**: 主动交易策略 vs 买入持有策略
- **性能指标**: 总收益率、夏普比率、最大回撤等

## 📁 项目结构

```
gradient-boosting-with-backtrader/
├── code_only.py              # 主程序文件
├── environment.yml           # Conda环境配置
├── README.md                 # 项目说明文档
├── downloaded_raw_data/      # 数据缓存目录
│   ├── 000300.SH_20041231_20250520.csv
│   └── 000300.SH_20041231_20250520.parquet
└── catboost_info/           # CatBoost训练日志
    ├── catboost_training.json
    ├── learn_error.tsv
    ├── time_left.tsv
    └── learn/
```

## 🔧 配置参数

主要配置参数在代码顶部可以调整：

```python
TICKER = '000300.SH'              # 股票代码
START_DATE = '2004-12-31'         # 开始日期
END_DATE = '2025-05-20'           # 结束日期
TEST_SIZE = 0.2                   # 测试集比例
POSITIVE_MOVE_THRESHOLD = 0.2     # 正向移动阈值(%)
RANDOM_STATE = 42                 # 随机种子
```

## 📈 性能指标

项目会输出以下性能指标：

- **分类指标**: 准确率、精确率、召回率、F1-score、AUC-ROC
- **回测指标**: 
  - 总收益率
  - 夏普比率
  - 最大回撤
  - 胜率
  - 年化收益率

## 🎯 模型预测目标

模型预测下一个交易日的价格变动：
- **标签1**: 下一日涨幅 > 0.2%
- **标签0**: 下一日涨幅 ≤ 0.2%

这是一个二分类问题，专注于识别短期上涨机会。

## 📚 技术栈

- **数据处理**: Pandas, NumPy
- **机器学习**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **数据可视化**: Matplotlib, Seaborn, Yellowbrick
- **统计分析**: Statsmodels
- **数据源**: Tushare

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📝 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## ⚠️ 免责声明

本项目仅用于教育和研究目的。任何基于此代码的投资决策风险自负。历史表现不代表未来结果。请在实际投资前咨询专业的财务顾问。

---

**开始你的量化投资之旅！** 🚀
