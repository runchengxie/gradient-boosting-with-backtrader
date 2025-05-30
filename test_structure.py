"""
测试脚本 - 验证模块化结构是否正常工作
"""

import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """测试所有模块是否可以正常导入"""
    try:
        # 测试配置模块
        from src.config import TICKER, START_DATE, END_DATE, POSITIVE_MOVE_THRESHOLD
        print("✅ 配置模块导入成功")
        print(f"   目标股票: {TICKER}")
        print(f"   时间范围: {START_DATE} 到 {END_DATE}")
        print(f"   正向移动阈值: {POSITIVE_MOVE_THRESHOLD}%")
        
        # 测试数据处理模块
        from src.data_processing import DataCollector, DataProcessor
        print("✅ 数据处理模块导入成功")
        
        # 测试特征工程模块
        from src.feature_engineering import FeatureEngineer
        print("✅ 特征工程模块导入成功")
        
        # 测试模型训练模块
        from src.model_training import ModelTrainer, ModelEvaluator
        print("✅ 模型训练模块导入成功")
        
        # 测试回测模块
        from src.backtesting import BacktestEngine
        print("✅ 回测模块导入成功")
        
        # 测试可视化模块
        from src.visualization import DataVisualizer
        print("✅ 可视化模块导入成功")
        
        # 测试工具模块
        from src.utils import print_data_info, chronological_split
        print("✅ 工具模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_functionality():
    """测试基本功能"""
    try:
        from src.data_processing import DataProcessor
        from src.feature_engineering import FeatureEngineer
        from src.model_training import ModelTrainer
        from src.backtesting import BacktestEngine
        from src.visualization import DataVisualizer
        
        # 测试类实例化
        processor = DataProcessor(positive_move_threshold=0.2)
        engineer = FeatureEngineer(random_state=42)
        trainer = ModelTrainer(random_state=42)
        backtest = BacktestEngine(initial_capital=100000)
        visualizer = DataVisualizer()
        
        print("✅ 所有类实例化成功")
        return True
        
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== 模块化结构测试 ===")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print()
    
    # 测试导入
    print("1. 测试模块导入...")
    import_success = test_imports()
    print()
    
    if import_success:
        # 测试功能
        print("2. 测试基本功能...")
        functionality_success = test_functionality()
        print()
        
        if functionality_success:
            print("🎉 所有测试通过！模块化结构工作正常")
            print()
            print("现在你可以运行:")
            print("  python main.py")
            print()
            print("或者单独使用模块:")
            print("  from src.data_processing import DataCollector")
            print("  from src.visualization import DataVisualizer")
        else:
            print("⚠️  模块导入成功，但功能测试失败")
    else:
        print("⚠️  模块导入失败，请检查依赖安装")

if __name__ == "__main__":
    main()
