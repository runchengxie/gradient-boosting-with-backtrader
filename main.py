"""
梯度提升股票预测项目 - 主程序
使用模块化结构重构后的版本
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入自定义模块
from src.config import *
from src.data_processing import DataCollector, DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer, ModelEvaluator
from src.backtesting import BacktestEngine
from src.backtrader_engine import BacktraderEngine  # 新增
from src.visualization import DataVisualizer
from src.utils import (
    print_data_info, print_feature_statistics, chronological_split,
    validate_data_quality, print_quality_report, save_results_to_file
)

# 忽略警告
warnings.filterwarnings('ignore')


def main():
    """主函数"""
    print("=== 梯度提升股票预测项目 ===")
    print(f"目标股票: {TICKER}")
    print(f"时间范围: {START_DATE} 到 {END_DATE}")
    print(f"正向移动阈值: {POSITIVE_MOVE_THRESHOLD}%")
    print("=" * 50)
    
    # 1. 数据收集
    print("\n步骤 1: 数据收集")
    collector = DataCollector()
    raw_data = collector.fetch_data(TICKER, START_DATE, END_DATE)
    print_data_info(raw_data, "原始数据")
    
    # 数据质量检查
    quality_report = validate_data_quality(raw_data)
    print_quality_report(quality_report)
      # 2. 数据可视化（初步）
    print("\n步骤 2: 数据可视化")
    visualizer = DataVisualizer(
        non_blocking=NON_BLOCKING_PLOTS,
        save_plots=SAVE_PLOTS
    )
    visualizer.plot_data_overview(raw_data)
    visualizer.plot_time_series(raw_data)
    visualizer.plot_correlation_matrix(raw_data)
    
    # 3. 数据处理
    print("\n步骤 3: 数据处理")
    processor = DataProcessor(positive_move_threshold=POSITIVE_MOVE_THRESHOLD)
    
    # 数据清洗
    cleaned_data = processor.clean_data(raw_data)
    print_data_info(cleaned_data, "清洗后数据")
    
    # 特征转换
    transformed_data = processor.apply_transformations(cleaned_data)
    print_data_info(transformed_data, "转换后数据")
    
    # 创建交互特征
    interaction_data = processor.create_interaction_features(transformed_data)
    print_data_info(interaction_data, "交互特征数据")
    
    # 创建目标变量
    data_with_target, target = processor.create_target_variable(interaction_data)
    visualizer.plot_target_distribution(target)
    
    # 准备特征矩阵
    X_full = processor.prepare_features(data_with_target)
    y_full = target
    print_feature_statistics(X_full, "完整特征集")
    
    # 4. 数据分割
    print("\n步骤 4: 数据分割")
    X_train_raw, X_test_raw, y_train, y_test = chronological_split(
        X_full, y_full, test_size=TEST_SIZE
    )
    
    # 5. 特征工程
    print("\n步骤 5: 特征工程")
    feature_engineer = FeatureEngineer(random_state=RANDOM_STATE)
    
    # 聚类特征
    X_train_cluster, X_test_cluster = feature_engineer.apply_clustering_features(
        X_train_raw, X_test_raw
    )
    
    # 特征选择
    X_train_final, X_test_final = feature_engineer.apply_feature_selection_pipeline(
        X_train_cluster, y_train, X_test_cluster
    )
    print_feature_statistics(X_train_final, "最终特征集")
    
    # 6. 模型训练
    print("\n步骤 6: 模型训练")
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    best_estimators = trainer.train_models(X_train_final, y_train)
    
    # 7. 模型评估
    print("\n步骤 7: 模型评估")
    evaluation_results = trainer.evaluate_models(X_test_final, y_test)
    
    # 模型比较
    evaluator = ModelEvaluator()
    comparison_df = evaluator.compare_models(evaluation_results)
    
    # 绘制ROC曲线比较
    trainer.plot_roc_curves(evaluation_results)
    
    # 选择最佳模型
    best_model_name, best_model = trainer.select_best_model(evaluation_results)
    
    if best_model:
        # 详细评估最佳模型
        evaluator.print_detailed_report(best_model_name, evaluation_results[best_model_name])
        
        # 可视化预测结果
        best_result = evaluation_results[best_model_name]
        visualizer.plot_prediction_distribution(
            y_test, best_result['prediction_probabilities']
        )
      # 8. 回测
    print("\n步骤 8: 策略回测")
    
    if USE_BACKTRADER:
        print("使用 Backtrader 进行高级回测")
        backtest_engine = BacktraderEngine(
            initial_capital=INITIAL_CAPITAL,
            commission=COMMISSION
        )
        
        # 准备回测数据
        test_data_for_backtest = data_with_target.loc[X_test_final.index].copy()
        
        # 买入持有策略
        backtest_engine.run_buy_and_hold_backtest(test_data_for_backtest)
        
        # 模型策略回测
        for model_name, result in evaluation_results.items():
            if result is not None:
                predictions = result['predictions']
                confidence_scores = result['prediction_probabilities']
                backtest_engine.run_strategy_backtest(
                    test_data_for_backtest, 
                    model_name, 
                    predictions,
                    confidence_scores
                )
        
        # 策略比较
        comparison_results = backtest_engine.compare_strategies()
        
        # 绘制结果
        if best_model_name:
            backtest_engine.plot_results(best_model_name)
            
    else:
        print("使用原始手写回测")
        backtest_engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
        
        # 准备回测数据
        test_data_for_backtest = data_with_target.loc[X_test_final.index].copy()
        
        # 买入持有策略
        backtest_engine.run_buy_and_hold_backtest(test_data_for_backtest)
        
        # 模型策略回测
        for model_name, result in evaluation_results.items():
            if result is not None:
                signals = result['predictions']
                backtest_engine.run_strategy_backtest(
                    test_data_for_backtest, model_name, signals
                )
        
        # 策略比较
        comparison_results = backtest_engine.compare_strategies()
        
        # 绘制结果
        backtest_engine.plot_portfolio_values()
        
        if best_model_name:
            backtest_engine.plot_drawdown(best_model_name)
    
    # 9. 保存结果
    print("\n步骤 9: 保存结果")
    results_summary = {
        'config': {
            'ticker': TICKER,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'test_size': TEST_SIZE,
            'positive_move_threshold': POSITIVE_MOVE_THRESHOLD
        },
        'data_info': {
            'raw_data_shape': raw_data.shape,
            'final_data_shape': X_train_final.shape,
            'target_distribution': target.value_counts().to_dict()
        },
        'model_performance': {
            name: {
                'auc': result['roc_auc'],
                'precision': result['classification_report']['1']['precision'],
                'recall': result['classification_report']['1']['recall'],
                'f1_score': result['classification_report']['1']['f1-score']
            }
            for name, result in evaluation_results.items()
            if result is not None
        },
        'best_model': best_model_name,
        'backtest_results': {
            name: result['performance_metrics']
            for name, result in backtest_engine.results.items()
        }
    }
    
    save_results_to_file(results_summary, 'results_summary.json')
    
    print("\n=== 项目完成 ===")
    print(f"最佳模型: {best_model_name}")
    if best_model_name and best_model_name in backtest_engine.results:
        best_backtest = backtest_engine.results[best_model_name]['performance_metrics']
        print(f"最佳模型回测收益: {best_backtest.get('total_return', 0):.2f}%")
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'evaluation_results': evaluation_results,
        'backtest_results': backtest_engine.results,
        'data': {
            'X_train': X_train_final,
            'X_test': X_test_final,
            'y_train': y_train,
            'y_test': y_test
        }
    }


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(RANDOM_STATE)
    
    try:
        results = main()
        print("\n程序执行成功！")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
