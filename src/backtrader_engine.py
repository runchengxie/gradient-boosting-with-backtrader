"""
基于Backtrader的高级回测模块
"""

import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import os


class MLSignalStrategy(bt.Strategy):
    """基于机器学习预测信号的策略"""
    
    params = (
        ('prediction_column', 'prediction'),
        ('confidence_threshold', 0.5),
        ('position_size', 0.95),  # 使用95%的资金
        ('stop_loss', 0.05),      # 5%止损
        ('take_profit', 0.10),    # 10%止盈
        ('predictions', None),    # 预测数组
        ('confidence_scores', None),  # 置信度数组
    )
    
    def __init__(self):
        self.order = None
        self.buy_price = None
        self.buy_date = None
          # 记录交易信息
        self.trades_log = []
    
    def next(self):
        """策略逻辑"""
        # 如果有未完成的订单，跳过
        if self.order:
            return
            
        current_idx = len(self.data) - 1
        
        # 检查是否有预测数据
        if (self.p.predictions is None or 
            current_idx >= len(self.p.predictions)):
            return
            
        current_prediction = self.p.predictions[current_idx]
        current_confidence = (self.p.confidence_scores[current_idx] 
                            if self.p.confidence_scores is not None 
                            else 1.0)
        
        # 当前持仓情况
        if not self.position:
            # 没有持仓，考虑买入
            if (current_prediction == 1 and 
                current_confidence >= self.p.confidence_threshold):
                
                # 计算买入数量
                size = int(self.broker.cash * self.p.position_size / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    self.buy_price = self.data.close[0]
                    self.buy_date = self.data.datetime.date(0)
                    
        else:
            # 有持仓，考虑卖出
            current_price = self.data.close[0]
            
            # 止损或止盈
            if self.buy_price is not None:
                pct_change = (current_price - self.buy_price) / self.buy_price
                
                if (pct_change <= -self.p.stop_loss or 
                    pct_change >= self.p.take_profit):
                    self.order = self.sell(size=self.position.size)
                    
            # 或者预测信号变为卖出
            elif current_prediction == 0:
                self.order = self.sell(size=self.position.size)
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: 价格={order.executed.price:.2f}, '
                        f'数量={order.executed.size}, '
                        f'费用={order.executed.comm:.2f}')
                self.buy_price = order.executed.price
            else:
                self.log(f'卖出执行: 价格={order.executed.price:.2f}, '
                        f'数量={order.executed.size}, '
                        f'费用={order.executed.comm:.2f}')
                
                # 记录完整交易
                if self.buy_price is not None:
                    profit = (order.executed.price - self.buy_price) * order.executed.size
                    profit_pct = (order.executed.price - self.buy_price) / self.buy_price * 100
                    
                    self.trades_log.append({
                        'buy_date': self.buy_date,
                        'sell_date': self.data.datetime.date(0),
                        'buy_price': self.buy_price,
                        'sell_price': order.executed.price,
                        'size': order.executed.size,
                        'profit': profit,
                        'profit_pct': profit_pct
                    })
                    
                self.buy_price = None
                self.buy_date = None
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单被拒绝/取消/保证金不足: {order.status}')
            
        self.order = None
    
    def log(self, txt, dt=None):
        """日志记录"""
        dt = dt or self.data.datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')


class BuyAndHoldStrategy(bt.Strategy):
    """买入并持有策略"""
    
    def __init__(self):
        self.order = None
        
    def next(self):
        if not self.position and not self.order:
            # 第一天买入
            size = int(self.broker.cash * 0.95 / self.data.close[0])
            if size > 0:
                self.order = self.buy(size=size)
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'买入并持有: 价格={order.executed.price:.2f}, '
                      f'数量={order.executed.size}')
        self.order = None


class BacktraderEngine:
    """Backtrader回测引擎"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
        
    def prepare_data(self, data: pd.DataFrame) -> bt.feeds.PandasData:
        """准备回测数据"""
        # 确保数据包含必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # 检查并创建缺失的列
        if 'open' not in data.columns and 'close' in data.columns:
            data['open'] = data['close'].shift(1).fillna(data['close'])
        if 'high' not in data.columns and 'close' in data.columns:
            data['high'] = data['close']
        if 'low' not in data.columns and 'close' in data.columns:
            data['low'] = data['close']
        if 'volume' not in data.columns:
            data['volume'] = 1000000  # 默认成交量
            
        # 重命名列以匹配backtrader期望的格式
        column_mapping = {
            'vol': 'volume',
            'vol_log': 'volume'
        }
        data = data.rename(columns=column_mapping)
        
        # 确保索引是DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
              # 选择必要的列
        bt_data = data[required_columns].copy()
        
        # 创建backtrader数据源
        data_feed = bt.feeds.PandasData(
            dataname=bt_data,
            datetime=None,  # 使用索引作为日期
            open=0, high=1, low=2, close=3, volume=4,
            openinterest=-1  # 不使用
        )
        
        return data_feed
    
    def run_strategy_backtest(self, data: pd.DataFrame, strategy_name: str,
                            predictions: np.ndarray, 
                            confidence_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """运行策略回测"""
        print(f"\n=== {strategy_name} Backtrader回测 ===")
        
        # 创建Cerebro引擎
        cerebro = bt.Cerebro()
        
        # 添加策略（在创建时传递预测数据）
        cerebro.addstrategy(MLSignalStrategy,
                          confidence_threshold=0.5,
                          position_size=0.95,
                          predictions=predictions,
                          confidence_scores=confidence_scores)
        
        # 准备数据
        data_feed = self.prepare_data(data)
        cerebro.adddata(data_feed)
        
        # 设置初始资金
        cerebro.broker.setcash(self.initial_capital)
        
        # 设置手续费
        cerebro.broker.setcommission(commission=self.commission)
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # 运行回测
        print('初始资金: %.2f' % cerebro.broker.getvalue())
        
        strategies = cerebro.run()
        strategy = strategies[0]
        
        final_value = cerebro.broker.getvalue()
        print('最终资金: %.2f' % final_value)
        
        # 获取分析结果
        returns_analyzer = strategy.analyzers.returns.get_analysis()
        sharpe_analyzer = strategy.analyzers.sharpe.get_analysis()
        drawdown_analyzer = strategy.analyzers.drawdown.get_analysis()
        trades_analyzer = strategy.analyzers.trades.get_analysis()
        
        # 计算性能指标
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        performance_metrics = {
            'total_return': total_return,
            'final_value': final_value,
            'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0),
            'max_drawdown': drawdown_analyzer.get('max', {}).get('drawdown', 0),
            'total_trades': trades_analyzer.get('total', {}).get('total', 0),
            'winning_trades': trades_analyzer.get('won', {}).get('total', 0),
            'losing_trades': trades_analyzer.get('lost', {}).get('total', 0),
            'win_rate': (trades_analyzer.get('won', {}).get('total', 0) / 
                        max(trades_analyzer.get('total', {}).get('total', 1), 1) * 100)
        }
        
        # 存储结果
        self.results[strategy_name] = {
            'performance_metrics': performance_metrics,
            'strategy': strategy,
            'cerebro': cerebro,
            'trades_log': getattr(strategy, 'trades_log', [])
        }
        
        # 打印结果
        self._print_backtest_results(strategy_name, performance_metrics)
        
        return self.results[strategy_name]
    
    def run_buy_and_hold_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """运行买入持有策略回测"""
        print(f"\n=== 买入持有策略 Backtrader回测 ===")
        
        # 创建Cerebro引擎
        cerebro = bt.Cerebro()
        
        # 添加策略
        cerebro.addstrategy(BuyAndHoldStrategy)
        
        # 准备数据
        data_feed = self.prepare_data(data)
        cerebro.adddata(data_feed)
        
        # 设置初始资金
        cerebro.broker.setcash(self.initial_capital)
        
        # 设置手续费
        cerebro.broker.setcommission(commission=self.commission)
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # 运行回测
        print('初始资金: %.2f' % cerebro.broker.getvalue())
        
        strategies = cerebro.run()
        strategy = strategies[0]
        
        final_value = cerebro.broker.getvalue()
        print('最终资金: %.2f' % final_value)
        
        # 获取分析结果
        returns_analyzer = strategy.analyzers.returns.get_analysis()
        sharpe_analyzer = strategy.analyzers.sharpe.get_analysis()
        drawdown_analyzer = strategy.analyzers.drawdown.get_analysis()
        trades_analyzer = strategy.analyzers.trades.get_analysis()
        
        # 计算性能指标
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        performance_metrics = {
            'total_return': total_return,
            'final_value': final_value,
            'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0),
            'max_drawdown': drawdown_analyzer.get('max', {}).get('drawdown', 0),
            'total_trades': trades_analyzer.get('total', {}).get('total', 0),
            'winning_trades': trades_analyzer.get('won', {}).get('total', 0),
            'losing_trades': trades_analyzer.get('lost', {}).get('total', 0),
            'win_rate': (trades_analyzer.get('won', {}).get('total', 0) / 
                        max(trades_analyzer.get('total', {}).get('total', 1), 1) * 100)
        }
        
        # 存储结果
        self.results['买入持有'] = {
            'performance_metrics': performance_metrics,
            'strategy': strategy,
            'cerebro': cerebro
        }
        
        # 打印结果
        self._print_backtest_results('买入持有', performance_metrics)
        
        return self.results['买入持有']
    
    def _print_backtest_results(self, strategy_name: str, metrics: Dict[str, float]):
        """打印回测结果"""
        print(f"\n{strategy_name} 回测结果:")
        print(f"总收益率: {metrics['total_return']:.2f}%")
        print(f"最终资金: {metrics['final_value']:.2f}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2f}%")
        print(f"总交易次数: {metrics['total_trades']}")
        print(f"胜率: {metrics['win_rate']:.2f}%")
        print("-" * 40)
    
    def compare_strategies(self) -> pd.DataFrame:
        """比较策略性能"""
        if not self.results:
            print("没有回测结果可比较")
            return pd.DataFrame()
        
        comparison_data = []
        for name, result in self.results.items():
            metrics = result['performance_metrics']
            comparison_data.append({
                '策略': name,
                '总收益率(%)': f"{metrics['total_return']:.2f}",
                '夏普比率': f"{metrics['sharpe_ratio']:.3f}",
                '最大回撤(%)': f"{metrics['max_drawdown']:.2f}",
                '总交易次数': metrics['total_trades'],
                '胜率(%)': f"{metrics['win_rate']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n策略比较:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_results(self, strategy_name: str = None):
        """绘制回测结果"""
        if not self.results:
            print("没有回测结果可绘制")
            return
            
        strategies_to_plot = [strategy_name] if strategy_name else list(self.results.keys())
        
        for name in strategies_to_plot:
            if name in self.results:
                print(f"\n绘制 {name} 策略结果...")
                cerebro = self.results[name]['cerebro']
                # 绘制资金曲线和交易信号
                cerebro.plot(style='candlestick', volume=False, figsize=(15, 10))
                plt.suptitle(f'{name} 策略回测结果')
                
                # 保存图片到plots文件夹
                os.makedirs('plots', exist_ok=True)
                filename = f'plots/backtrader_{name.replace(" ", "_").lower()}_results.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"图表已保存: {filename}")
                
                plt.show(block=False)
                plt.pause(0.1)
