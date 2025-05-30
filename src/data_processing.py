"""
数据获取和处理模块
"""

import os
import pandas as pd
import numpy as np
import tushare as ts
from typing import Optional, Tuple
from .config import CACHE_DIR


class DataCollector:
    """数据收集类"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('TUSHARE_API_KEY')
        if self.api_key:
            ts.set_token(self.api_key)
            self.pro = ts.pro_api()
        
    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票/指数数据"""
        cache_file_name = f"{ticker}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
        cache_file_path = os.path.join(CACHE_DIR, cache_file_name)
        
        # 创建缓存目录
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        
        # 检查缓存
        if os.path.exists(cache_file_path):
            print(f"从缓存加载数据: {cache_file_path}")
            return pd.read_parquet(cache_file_path)
        
        # 从API获取数据
        print(f"从Tushare API获取数据: {ticker}")
        if not self.api_key:
            raise ValueError("TUSHARE_API_KEY 环境变量未设置")
        
        data = self._fetch_from_api(ticker, start_date, end_date)
        
        # 保存到缓存
        try:
            data.to_parquet(cache_file_path)
            print(f"数据已保存到缓存: {cache_file_path}")
        except Exception as e:
            print(f"保存缓存失败: {e}")
        
        return data
    
    def _fetch_from_api(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从API获取数据的内部方法"""
        start_date_ts = pd.to_datetime(start_date).strftime('%Y%m%d')
        end_date_ts = pd.to_datetime(end_date).strftime('%Y%m%d')
        
        factor_fields = [
            'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'pct_change',
            'ma_bfq_5', 'ma_bfq_20', 'ma_bfq_60', 'ema_bfq_10', 'ema_bfq_60',
            'macd_dif_bfq', 'macd_dea_bfq', 'macd_bfq', 'dmi_adx_bfq', 'dmi_pdi_bfq',
            'dmi_mdi_bfq', 'roc_bfq', 'mtm_bfq', 'updays', 'downdays', 'atr_bfq',
            'boll_upper_bfq', 'boll_lower_bfq', 'rsi_bfq_12', 'cci_bfq', 'wr_bfq',
            'kdj_k_bfq', 'kdj_d_bfq', 'obv_bfq', 'mfi_bfq'
        ]
        
        data = self.pro.idx_factor_pro(
            ts_code=ticker,
            start_date=start_date_ts,
            end_date=end_date_ts,
            fields=",".join(factor_fields)
        )
        
        if data.empty:
            raise ValueError(f"未找到股票代码 {ticker} 的数据")
        
        data['trade_date'] = pd.to_datetime(data['trade_date'])
        data.set_index('trade_date', inplace=True)
        data.sort_index(inplace=True)
        
        return data


class DataProcessor:
    """数据处理类"""
    
    def __init__(self, positive_move_threshold: float = 0.2):
        self.positive_move_threshold = positive_move_threshold
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        processed_data = data.copy()
        
        # 检查必要列
        if 'pct_change' not in processed_data.columns:
            raise ValueError("数据中缺少 'pct_change' 列")
        
        # 价格合理性检查
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in processed_data.columns:
                if (processed_data[col] <= 0).any():
                    raise ValueError(f"列 '{col}' 包含非正值")
        
        # OHLC逻辑关系检查
        self._validate_ohlc_logic(processed_data)
        
        # 删除NaN值
        processed_data.dropna(inplace=True)
        
        print(f"数据清洗后形状: {processed_data.shape}")
        return processed_data
    
    def _validate_ohlc_logic(self, data: pd.DataFrame):
        """验证OHLC数据的逻辑关系"""
        checks = [
            ('high', 'open', '>='),
            ('high', 'close', '>='),
            ('low', 'open', '<='),
            ('low', 'close', '<=')
        ]
        
        for col1, col2, op in checks:
            if col1 in data.columns and col2 in data.columns:
                if op == '>=' and not (data[col1] >= data[col2]).all():
                    raise ValueError(f"数据完整性问题: {col1} 应该 >= {col2}")
                elif op == '<=' and not (data[col1] <= data[col2]).all():
                    raise ValueError(f"数据完整性问题: {col1} 应该 <= {col2}")
    
    def apply_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用数据转换"""
        processed_data = data.copy()
        
        # 对数转换
        if 'vol' in processed_data.columns:
            processed_data['vol_log'] = np.log1p(processed_data['vol'])
            processed_data.drop(columns=['vol'], inplace=True)
        
        if 'amount' in processed_data.columns:
            processed_data['amount_log'] = np.log1p(processed_data['amount'])
            processed_data.drop(columns=['amount'], inplace=True)
        
        # 差分转换
        if 'obv_bfq' in processed_data.columns:
            processed_data['obv_bfq_diff'] = processed_data['obv_bfq'].diff()
            processed_data.drop(columns=['obv_bfq'], inplace=True)
        
        # 删除转换后产生的NaN
        processed_data.dropna(inplace=True)
        
        print(f"特征转换后形状: {processed_data.shape}")
        return processed_data
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建交互特征"""
        processed_data = data.copy()
        
        # 趋势与动量交互
        if 'ma_bfq_20' in processed_data.columns and 'rsi_bfq_12' in processed_data.columns:
            processed_data['inter_ma20_rsi12'] = processed_data['ma_bfq_20'] * processed_data['rsi_bfq_12']
        
        # 趋势与波动性交互
        if 'ma_bfq_20' in processed_data.columns and 'atr_bfq' in processed_data.columns:
            processed_data['inter_ma20_div_atr'] = processed_data['ma_bfq_20'] / (processed_data['atr_bfq'] + 1e-6)
        
        # 动量指标交互
        if 'rsi_bfq_12' in processed_data.columns and 'mfi_bfq' in processed_data.columns:
            processed_data['inter_rsi12_mfi'] = processed_data['rsi_bfq_12'] * processed_data['mfi_bfq']
        
        # 特定模式组合
        if all(col in processed_data.columns for col in ['close', 'boll_lower_bfq', 'rsi_bfq_12']):
            processed_data['inter_boll_rsi_pattern'] = (
                (processed_data['close'] - processed_data['boll_lower_bfq']) * 
                (processed_data['rsi_bfq_12'] < 30).astype(int)
            )
        
        processed_data.dropna(inplace=True)
        print(f"交互特征创建后形状: {processed_data.shape}")
        return processed_data
    
    def create_target_variable(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """创建目标变量"""
        data_with_target = data.copy()
        
        # 预测下一天的涨跌
        data_with_target['pct_change_next_day'] = data_with_target['pct_change'].shift(-1)
        data_with_target.dropna(subset=['pct_change_next_day'], inplace=True)
        
        # 定义目标变量
        data_with_target['Target'] = np.where(
            data_with_target['pct_change_next_day'] > self.positive_move_threshold, 1, 0
        )
        
        print(f"目标变量分布:")
        print(data_with_target['Target'].value_counts(normalize=True))
        
        return data_with_target, data_with_target['Target']
    
    def prepare_features(self, data_with_target: pd.DataFrame) -> pd.DataFrame:
        """准备特征矩阵"""
        # 排除目标相关列和原始价格数据
        exclude_columns = [
            'Target', 'pct_change_next_day', 'close', 'open', 'high', 'low'
        ]
        
        potential_features = [
            col for col in data_with_target.columns 
            if col not in exclude_columns
        ]
        
        return data_with_target[potential_features].copy()
