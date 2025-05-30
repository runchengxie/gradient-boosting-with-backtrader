"""
特征工程模块
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from yellowbrick.cluster import KElbowVisualizer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Tuple, List, Optional


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        self.feature_selector = None
    
    def apply_clustering_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """应用聚类特征工程"""
        if X_train.shape[0] <= 5:
            print("样本数量太少，跳过聚类特征工程")
            return X_train, X_test
        
        # 标准化数据用于聚类
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 确定最优K值
        optimal_k = self._find_optimal_k(X_train_scaled)
        
        # 应用K-means聚类
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init='auto')
        train_clusters = self.kmeans.fit_predict(X_train_scaled)
        
        # 为测试集预测聚类
        X_test_scaled = self.scaler.transform(X_test)
        test_clusters = self.kmeans.predict(X_test_scaled)
        
        # 添加聚类特征
        X_train_with_clusters = X_train.copy()
        X_test_with_clusters = X_test.copy()
        
        X_train_with_clusters['Cluster'] = train_clusters
        X_test_with_clusters['Cluster'] = test_clusters
        
        print(f"添加聚类特征，使用 {optimal_k} 个聚类")
        print(f"训练集聚类分布: {pd.Series(train_clusters).value_counts().sort_index()}")
        
        return X_train_with_clusters, X_test_with_clusters
    
    def _find_optimal_k(self, X_scaled: np.ndarray, k_range: Tuple[int, int] = (2, 11)) -> int:
        """使用肘部法则找到最优K值"""
        try:
            model_elbow = KMeans(random_state=self.random_state, n_init='auto')
            visualizer_elbow = KElbowVisualizer(model_elbow, k=k_range, metric='distortion', timings=False)
            visualizer_elbow.fit(X_scaled)
            optimal_k = visualizer_elbow.elbow_value_
            
            if optimal_k is None or optimal_k < 2:
                print("肘部法则未能确定明确的K值，使用默认值3")
                return 3
            
            return optimal_k
        except Exception as e:
            print(f"确定最优K值时出错: {e}，使用默认值3")
            return 3
    
    def select_features_univariate(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                 k: int = 20) -> List[str]:
        """单变量特征选择"""
        try:
            selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
            selector.fit(X_train, y_train)
            
            # 获取选中的特征
            feature_scores = pd.DataFrame({
                'feature': X_train.columns,
                'score': selector.scores_,
                'selected': selector.get_support()
            })
            
            selected_features = feature_scores[feature_scores['selected']]['feature'].tolist()
            
            print(f"单变量特征选择: 从 {X_train.shape[1]} 个特征中选择了 {len(selected_features)} 个")
            print("前10个特征得分:")
            print(feature_scores.nlargest(10, 'score')[['feature', 'score']])
            
            return selected_features
        except Exception as e:
            print(f"单变量特征选择失败: {e}")
            return X_train.columns.tolist()
    
    def select_features_rfe(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           n_features: int = 15) -> List[str]:
        """递归特征消除"""
        try:
            estimator = LogisticRegression(random_state=self.random_state, max_iter=1000)
            n_features = min(n_features, X_train.shape[1])
            
            rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
            rfe.fit(X_train, y_train)
            
            selected_features = X_train.columns[rfe.support_].tolist()
            
            # 特征排名
            feature_ranking = pd.DataFrame({
                'feature': X_train.columns,
                'ranking': rfe.ranking_,
                'selected': rfe.support_
            }).sort_values('ranking')
            
            print(f"RFE特征选择: 选择了 {len(selected_features)} 个特征")
            print("特征排名前10:")
            print(feature_ranking.head(10))
            
            return selected_features
        except Exception as e:
            print(f"RFE特征选择失败: {e}")
            return X_train.columns.tolist()
    
    def check_multicollinearity(self, X: pd.DataFrame, threshold: float = 10.0) -> List[str]:
        """检查多重共线性并返回需要移除的特征"""
        try:
            # 只对数值型特征计算VIF
            numeric_features = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_features]
            
            if X_numeric.shape[1] < 2:
                return []
            
            # 计算VIF
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X_numeric.columns
            vif_data["VIF"] = [
                variance_inflation_factor(X_numeric.values, i) 
                for i in range(X_numeric.shape[1])
            ]
            
            # 找出VIF过高的特征
            high_vif_features = vif_data[vif_data["VIF"] > threshold]["Feature"].tolist()
            
            print(f"VIF分析完成，阈值: {threshold}")
            print("VIF值排序:")
            print(vif_data.sort_values("VIF", ascending=False))
            
            if high_vif_features:
                print(f"发现 {len(high_vif_features)} 个高VIF特征: {high_vif_features}")
            
            return high_vif_features
        except Exception as e:
            print(f"VIF分析失败: {e}")
            return []
    
    def _progressive_vif_removal(self, X: pd.DataFrame, features: List[str], threshold: float = 15.0) -> List[str]:
        """渐进式移除高VIF特征"""
        try:
            remaining_features = features.copy()
            X_current = X[remaining_features]
            
            while True:
                # 只对数值型特征计算VIF
                numeric_features = X_current.select_dtypes(include=[np.number]).columns
                X_numeric = X_current[numeric_features]
                
                if X_numeric.shape[1] < 2:
                    break
                
                # 计算当前特征的VIF
                vif_data = pd.DataFrame()
                vif_data["Feature"] = X_numeric.columns
                vif_data["VIF"] = [
                    variance_inflation_factor(X_numeric.values, i) 
                    for i in range(X_numeric.shape[1])
                ]
                
                # 找出VIF最高的特征
                max_vif_row = vif_data.loc[vif_data["VIF"].idxmax()]
                max_vif_feature = max_vif_row["Feature"]
                max_vif_value = max_vif_row["VIF"]
                
                # 如果最高VIF低于阈值，停止移除
                if max_vif_value <= threshold:
                    break
                
                # 移除VIF最高的特征
                remaining_features.remove(max_vif_feature)
                X_current = X_current.drop(columns=[max_vif_feature])
                print(f"移除高VIF特征: {max_vif_feature} (VIF: {max_vif_value:.2f})")
                
                # 如果特征数量过少，停止移除
                if len(remaining_features) <= 5:
                    print(f"特征数量已降至 {len(remaining_features)}，停止移除")
                    break
            
            print(f"VIF移除完成，保留 {len(remaining_features)} 个特征")
            return remaining_features
            
        except Exception as e:
            print(f"渐进式VIF移除失败: {e}")
            return features
    
    def select_features_embedded(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                n_features_ratio: float = 0.8) -> List[str]:
        """嵌入式特征选择（基于梯度提升的特征重要性）"""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            
            # 使用简单的梯度提升模型获取特征重要性
            gb_model = GradientBoostingClassifier(
                random_state=self.random_state, 
                n_estimators=50, 
                max_depth=3
            )
            gb_model.fit(X_train, y_train)
            
            # 获取特征重要性
            importances = pd.Series(gb_model.feature_importances_, index=X_train.columns)
            
            if not importances.empty:
                n_features = max(3, int(len(importances) * n_features_ratio))
                if n_features > len(importances):
                    n_features = len(importances)
                
                selected_features = importances.nlargest(n_features).index.tolist()
                
                print(f"嵌入式特征选择: 从 {X_train.shape[1]} 个特征中选择了 {len(selected_features)} 个")
                print("前10个重要特征:")
                top_features = importances.nlargest(10)
                for feature, importance in top_features.items():
                    print(f"  {feature}: {importance:.4f}")
                
                return selected_features
            else:
                print("嵌入式特征选择失败: 无法获取特征重要性")
                return X_train.columns.tolist()
                
        except Exception as e:
            print(f"嵌入式特征选择失败: {e}")
            return X_train.columns.tolist()
    
    def apply_feature_selection_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series,
                                       X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """应用完整的特征选择流水线（与reference_code一致的顺序）"""
        print("\n=== 开始特征选择流水线（Reference Code风格）===")
        print(f"初始特征数: {X_train.shape[1]}")
        
        # 1. VIF过滤（阈值10.0）
        print("\n1. VIF过滤")
        vif_features = self._progressive_vif_removal(X_train, X_train.columns.tolist(), threshold=10.0)
        X_train_vif = X_train[vif_features]
        X_test_vif = X_test[vif_features]
        print(f"VIF过滤后保留特征数: {len(vif_features)}")
        
        # 2. 单变量特征选择（保留75%特征）
        print("\n2. 单变量特征选择（SelectKBest）")
        k_filter = max(10, int(X_train_vif.shape[1] * 0.75))
        if k_filter > X_train_vif.shape[1]:
            k_filter = X_train_vif.shape[1]
        univariate_features = self.select_features_univariate(X_train_vif, y_train, k=k_filter)
        X_train_uni = X_train_vif[univariate_features]
        X_test_uni = X_test_vif[univariate_features]
        print(f"单变量特征选择后保留特征数: {len(univariate_features)}")
        
        # 3. RFE特征选择（保留67%特征）
        print("\n3. 递归特征消除（RFE）")
        n_wrapper = max(5, int(X_train_uni.shape[1] * 0.67))
        if n_wrapper > X_train_uni.shape[1]:
            n_wrapper = X_train_uni.shape[1]
        rfe_features = self.select_features_rfe(X_train_uni, y_train, n_features=n_wrapper)
        X_train_rfe = X_train_uni[rfe_features]
        X_test_rfe = X_test_uni[rfe_features]
        print(f"RFE后保留特征数: {len(rfe_features)}")
        
        # 4. 嵌入式特征选择（保留80%特征）
        print("\n4. 嵌入式特征选择")
        final_features = self.select_features_embedded(X_train_rfe, y_train, n_features_ratio=0.8)
        X_train_final = X_train_rfe[final_features]
        X_test_final = X_test_rfe[final_features]
        
        print(f"\n特征选择完成:")
        print(f"原始特征数: {X_train.shape[1]}")
        print(f"最终特征数: {len(final_features)}")
        print(f"最终特征: {final_features}")
        
        # 打印最终特征的统计信息
        print(f"\n=== 最终特征集 统计信息 ===")
        print(f"特征数量: {len(final_features)}")
        print(f"样本数量: {X_train_final.shape[0]}")
        print("特征列表:")
        for i, feature in enumerate(final_features, 1):
            print(f"   {i}. {feature}")
        
        return X_train_final, X_test_final
