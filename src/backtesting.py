"""
回测模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Any, Optional


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = {}
    
    def run_strategy_backtest(self, data: pd.DataFrame, model_name: str, 
                            signals: np.ndarray) -> Dict[str, Any]:
        """运行策略回测"""
        print(f"\n=== {model_name} 策略回测 ===")
        
        # 准备数据
        backtest_data = data.copy()
        backtest_data['Predicted_Signal'] = signals
        
        # 执行回测
        portfolio_values, trade_log = self._execute_backtest(backtest_data)
        
        # 计算性能指标
        performance_metrics = self._calculate_performance_metrics(portfolio_values)
        
        # 存储结果
        self.results[model_name] = {
            'portfolio_values': portfolio_values,
            'trade_log': trade_log,
            'performance_metrics': performance_metrics,
            'dates': backtest_data.index[:len(portfolio_values)]
        }
        
        # 打印结果
        self._print_backtest_results(model_name, performance_metrics, trade_log)
        
        return self.results[model_name]
    
    def _execute_backtest(self, data: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """执行回测逻辑"""
        capital = self.initial_capital
        shares_held = 0
        portfolio_values = []
        trade_log = []
        
        for date, row in data.iterrows():
            current_price = row['close']
            signal = row['Predicted_Signal']
            
            # 买入信号
            if signal == 1 and shares_held == 0:
                shares_to_buy = int(capital // current_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    capital -= cost
                    shares_held += shares_to_buy
                    trade_log.append(f"{date.strftime('%Y-%m-%d')}: 买入 {shares_to_buy} 股，价格 {current_price:.2f}，成本 {cost:.2f}")
            
            # 卖出信号
            elif signal == 0 and shares_held > 0:
                proceeds = shares_held * current_price
                capital += proceeds
                trade_log.append(f"{date.strftime('%Y-%m-%d')}: 卖出 {shares_held} 股，价格 {current_price:.2f}，收益 {proceeds:.2f}")
                shares_held = 0
            
            # 计算当前组合价值
            current_portfolio_value = capital + (shares_held * current_price)
            portfolio_values.append(current_portfolio_value)
        
        return portfolio_values, trade_log
    
    def run_buy_and_hold_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """运行买入持有策略回测"""
        print("\n=== 买入持有策略回测 ===")
        
        if data.empty or 'close' not in data.columns:
            return None
        
        first_price = data['close'].iloc[0]
        shares_bought = int(self.initial_capital // first_price)
        remaining_cash = self.initial_capital - (shares_bought * first_price)
        
        portfolio_values = []
        for _, row in data.iterrows():
            current_price = row['close']
            current_value = remaining_cash + (shares_bought * current_price)
            portfolio_values.append(current_value)
        
        performance_metrics = self._calculate_performance_metrics(portfolio_values)
        
        result = {
            'portfolio_values': portfolio_values,
            'performance_metrics': performance_metrics,
            'dates': data.index[:len(portfolio_values)],
            'shares_bought': shares_bought,
            'remaining_cash': remaining_cash
        }
        
        self.results['Buy_and_Hold'] = result
        
        print(f"买入 {shares_bought} 股，价格 {first_price:.2f}")
        print(f"剩余现金: {remaining_cash:.2f}")
        self._print_performance_metrics('买入持有', performance_metrics)
        
        return result
    
    def _calculate_performance_metrics(self, portfolio_values: List[float]) -> Dict[str, float]:
        """计算性能指标"""
        if not portfolio_values:
            return {}
        
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # 计算年化收益率
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        annual_return = np.mean(returns) * 252 * 100  # 假设252个交易日
        
        # 计算波动率
        volatility = np.std(returns) * np.sqrt(252) * 100
        
        # 计算夏普比率
        risk_free_rate = 0.02  # 假设无风险利率2%
        sharpe_ratio = (annual_return - risk_free_rate * 100) / volatility if volatility > 0 else 0
        
        # 计算最大回撤
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': final_value
        }
    
    def _print_backtest_results(self, model_name: str, metrics: Dict[str, float], 
                              trade_log: List[str]):
        """打印回测结果"""
        print(f"初始资金: {self.initial_capital:,.2f}")
        self._print_performance_metrics(model_name, metrics)
        
        print(f"\n交易记录 (总共 {len(trade_log)} 笔交易):")
        if trade_log:
            for i, trade in enumerate(trade_log[:10]):  # 只显示前10笔交易
                print(f"  {trade}")
            if len(trade_log) > 10:
                print(f"  ... 还有 {len(trade_log) - 10} 笔交易")
        else:
            print("  无交易记录")
    
    def _print_performance_metrics(self, strategy_name: str, metrics: Dict[str, float]):
        """打印性能指标"""
        print(f"\n{strategy_name} 策略表现:")
        for metric, value in metrics.items():
            if metric == 'final_value':
                print(f"  最终价值: {value:,.2f}")
            elif 'return' in metric or 'volatility' in metric or 'drawdown' in metric:
                print(f"  {metric}: {value:.2f}%")
            else:
                print(f"  {metric}: {value:.4f}")
    
    def compare_strategies(self):
        """比较所有策略"""
        if not self.results:
            print("没有回测结果可比较")
            return
        
        print("\n=== 策略比较 ===")
        comparison_data = []
        
        for strategy_name, result in self.results.items():
            metrics = result['performance_metrics']
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return (%)': metrics.get('total_return', 0),
                'Annual Return (%)': metrics.get('annual_return', 0),
                'Volatility (%)': metrics.get('volatility', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Max Drawdown (%)': metrics.get('max_drawdown', 0),
                'Final Value': metrics.get('final_value', 0)
            })
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Total Return (%)', ascending=False)
        
        print(comparison_df.round(4))
        return comparison_df
    
    def plot_portfolio_values(self):
        """绘制组合价值走势"""
        if not self.results:
            print("没有回测结果可绘制")
            return
        
        plt.figure(figsize=(14, 8))
        
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
        linestyles = ['-', '--', '-.', ':']
        
        for i, (strategy_name, result) in enumerate(self.results.items()):
            dates = result['dates']
            portfolio_values = result['portfolio_values']
            
            if portfolio_values and dates is not None:
                color = colors[i % len(colors)]
                linestyle = linestyles[i % len(linestyles)]
                
                plt.plot(dates, portfolio_values[:len(dates)], 
                        label=f'{strategy_name}', 
                        color=color, 
                        linestyle=linestyle,
                        linewidth=2)
        
        plt.title('组合价值走势比较')
        plt.xlabel('日期')
        plt.ylabel(f'组合价值 (初始资金: {self.initial_capital:,.0f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图片到plots文件夹
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/portfolio_values_comparison.png', dpi=300, bbox_inches='tight')
        print("图表已保存: plots/portfolio_values_comparison.png")
        
        plt.show(block=False)
        plt.pause(0.1)
    
    def plot_drawdown(self, strategy_name: str):
        """绘制回撤图"""
        if strategy_name not in self.results:
            print(f"策略 {strategy_name} 的回测结果不存在")
            return
        
        result = self.results[strategy_name]
        portfolio_values = result['portfolio_values']
        dates = result['dates']
        
        if not portfolio_values:
            print("没有组合价值数据")
            return
          # 计算回撤
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        plt.plot(dates, drawdown, color='red', linewidth=1)
        plt.title(f'{strategy_name} 策略回撤图')
        plt.xlabel('日期')
        plt.ylabel('回撤 (%)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图片到plots文件夹
        os.makedirs('plots', exist_ok=True)
        filename = f'plots/drawdown_{strategy_name.replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {filename}")
        
        plt.show(block=False)
        plt.pause(0.1)
