"""
模型训练和评估模块
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import os


class ModelTrainer:
    """模型训练类"""
    
    def __init__(self, random_state: int = 42, cv_folds: int = 3):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.best_estimators = {}
        self.cv_results = {}
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取模型配置"""
        return {
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                }
            },
            'LightGBM': {
                'model': LGBMClassifier(random_state=self.random_state, verbose=-1),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                }
            },
            'CatBoost': {
                'model': CatBoostClassifier(random_state=self.random_state, verbose=False),
                'params': {
                    'iterations': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'depth': [3, 5]
                }
            }
        }
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """训练所有模型"""
        print("=== 开始模型训练 ===")
        
        # 标准化数据
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        model_configs = self.get_model_configs()
        
        for model_name, config in model_configs.items():
            print(f"\n训练 {model_name}...")
            
            try:
                # 网格搜索
                grid_search = GridSearchCV(
                    estimator=config['model'],
                    param_grid=config['params'],
                    cv=tscv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train_scaled, y_train)
                
                self.best_estimators[model_name] = grid_search.best_estimator_
                self.cv_results[model_name] = {
                    'best_score': grid_search.best_score_,
                    'best_params': grid_search.best_params_,
                    'cv_results': grid_search.cv_results_
                }
                
                print(f"{model_name} 训练完成")
                print(f"最佳CV AUC: {grid_search.best_score_:.4f}")
                print(f"最佳参数: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"{model_name} 训练失败: {e}")
                self.best_estimators[model_name] = None
                self.cv_results[model_name] = None
        
        return self.best_estimators
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """评估所有模型"""
        print("\n=== 开始模型评估 ===")
        
        # 标准化测试数据
        X_test_scaled = self.scaler.transform(X_test)
        
        evaluation_results = {}
        all_roc_data = {}
        
        for model_name, model in self.best_estimators.items():
            if model is None:
                print(f"跳过 {model_name} (训练失败)")
                continue
            
            print(f"\n评估 {model_name}...")
            
            # 预测
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
            
            # 计算指标
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            evaluation_results[model_name] = {
                'roc_auc': roc_auc,
                'confusion_matrix': cm,
                'classification_report': report,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba,
                'fpr': fpr,
                'tpr': tpr
            }
            
            all_roc_data[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
            
            print(f"{model_name} 测试集 AUC: {roc_auc:.4f}")
          # 绘制ROC曲线
        self.plot_roc_curves(evaluation_results)
        
        return evaluation_results
    
    def plot_roc_curves(self, evaluation_results: Dict[str, Dict[str, Any]]):
        """绘制ROC曲线比较"""
        plt.figure(figsize=(10, 8))
        
        roc_data = {}
        for model_name, result in evaluation_results.items():
            if result is not None:
                roc_data[model_name] = {
                    'fpr': result['fpr'],
                    'tpr': result['tpr'],
                    'auc': result['roc_auc']
                }
        
        for model_name, data in roc_data.items():
            plt.plot(data['fpr'], data['tpr'], lw=2,
                    label=f'{model_name} (AUC = {data["auc"]:.2f})')
        
        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title('ROC曲线比较 - 测试集')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        # 保存图片到plots文件夹
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        print("图表已保存: plots/roc_curves_comparison.png")
        
        plt.show(block=False)
        plt.pause(0.1)
    
    def select_best_model(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Tuple[str, Any]:
        """选择最佳模型"""
        if not evaluation_results:
            return "", None
        
        # 基于AUC选择最佳模型
        best_model_name = max(evaluation_results.keys(), 
                            key=lambda x: evaluation_results[x]['roc_auc'])
        best_model = self.best_estimators[best_model_name]
        
        print(f"\n=== 最佳模型: {best_model_name} ===")
        print(f"测试集 AUC: {evaluation_results[best_model_name]['roc_auc']:.4f}")
          # 绘制最佳模型的混淆矩阵
        self._plot_confusion_matrix(
            evaluation_results[best_model_name]['confusion_matrix'],
            best_model_name
        )
        
        return best_model_name, best_model
    
    def _plot_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """绘制混淆矩阵"""
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['预测负向', '预测正向'],
                   yticklabels=['实际负向', '实际正向'])
        plt.title(f'混淆矩阵 - {model_name}')
        plt.ylabel('实际类别')
        plt.xlabel('预测类别')
        
        # 保存图片到plots文件夹
        os.makedirs('plots', exist_ok=True)
        filename = f'plots/confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {filename}")
        
        plt.show(block=False)
        plt.pause(0.1)


class ModelEvaluator:
    """模型评估器"""
    
    @staticmethod
    def print_detailed_report(model_name: str, evaluation_result: Dict[str, Any]):
        """打印详细的评估报告"""
        print(f"\n=== {model_name} 详细评估报告 ===")
        
        print(f"AUC得分: {evaluation_result['roc_auc']:.4f}")
        
        print("\n混淆矩阵:")
        print(evaluation_result['confusion_matrix'])
        
        print("\n分类报告:")
        report = evaluation_result['classification_report']
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"{class_name}:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
    
    @staticmethod
    def compare_models(evaluation_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """比较所有模型的性能"""
        comparison_data = []
        
        for model_name, result in evaluation_results.items():
            report = result['classification_report']
            
            comparison_data.append({
                'Model': model_name,
                'AUC': result['roc_auc'],
                'Precision': report['1']['precision'],
                'Recall': report['1']['recall'],
                'F1-Score': report['1']['f1-score'],
                'Accuracy': report['accuracy']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC', ascending=False)
        
        print("\n=== 模型性能比较 ===")
        print(comparison_df.round(4))
        
        return comparison_df
