"""
配置文件 - 包含所有项目配置和常量
"""

# 数据相关配置
CACHE_DIR = 'downloaded_raw_data'
TICKER = '000300.SH'  # 沪深300指数
START_DATE = '2004-12-31'
END_DATE = '2025-05-20'

# 模型相关配置
TEST_SIZE = 0.2  # 测试集比例
RANDOM_STATE = 42
POSITIVE_MOVE_THRESHOLD = 0.2  # 定义"正向移动"的阈值

# 回测相关配置
INITIAL_CAPITAL = 100000  # 初始资金
USE_BACKTRADER = True     # 是否使用Backtrader进行回测（False使用原始手写回测）
COMMISSION = 0.001        # 手续费率

# 可视化配置
FIGURE_SIZE_SMALL = (10, 6)
FIGURE_SIZE_MEDIUM = (12, 8)
FIGURE_SIZE_LARGE = (15, 10)
NON_BLOCKING_PLOTS = True  # 是否使用非阻塞图表显示
SAVE_PLOTS = True         # 是否保存图表到文件
