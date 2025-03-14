import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.patterns.pattern_factory import PatternFactory

logger = logging.getLogger('BacktestEngine')

class BacktestEngine:
    """
    Engine for backtesting harmonic patterns on historical data.
    """
    
    def __init__(self, data, initial_capital=10000, position_size=0.1, stop_loss=0.02, take_profit=0.05):
        """
        Initialize the backtest engine.
        
        Args:
            data (pandas.DataFrame): DataFrame with OHLCV data
            initial_capital (float): Initial capital for backtesting
            position_size (float): Position size as a percentage of capital
            stop_loss (float): Stop loss percentage
            take_profit (float): Take profit percentage
        """
        self.data = data
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.pattern_factory = PatternFactory()
        
        logger.info(f"Backtest engine initialized with {len(data)} data points")
    
    def run_backtest(self, pattern_names=None, tolerance=0.05, swing_window=5):
        """
        Run backtest for specified patterns.
        
        Args:
            pattern_names (list, optional): List of pattern names to test. If None, test all patterns.
            tolerance (float): Tolerance for Fibonacci ratio matching
            swing_window (int): Window size for detecting swings
            
        Returns:
            dict: Dictionary of pattern name to performance metrics
        """
        if pattern_names is None:
            pattern_names = self.pattern_factory.list_available_patterns()
        
        results = {}
        
        for pattern_name in pattern_names:
            logger.info(f"Running backtest for {pattern_name} pattern")
            
            try:
                pattern = self.pattern_factory.get_pattern(pattern_name, tolerance)
                patterns_found = pattern.find_patterns(self.data, swing_window)
                
                if not patterns_found:
                    logger.warning(f"No {pattern_name} patterns found in the data")
                    results[pattern_name] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'win_rate': 0,
                        'avg_profit': 0,
                        'avg_loss': 0,
                        'profit_factor': 0,
                        'total_return': 0,
                        'patterns_found': 0
                    }
                    continue
                
                performance = pattern.evaluate_performance(
                    self.data, 
                    patterns_found, 
                    stop_loss=self.stop_loss, 
                    take_profit=self.take_profit
                )
                
                performance['patterns_found'] = len(patterns_found)
                results[pattern_name] = performance
                
                logger.info(f"Backtest for {pattern_name} completed: {len(patterns_found)} patterns found, "
                           f"win rate: {performance['win_rate']:.2%}, total return: {performance['total_return']:.2%}")
                
            except Exception as e:
                logger.error(f"Error running backtest for {pattern_name}: {e}")
                results[pattern_name] = {
                    'error': str(e)
                }
        
        return results
    
    def plot_patterns(self, pattern_name, patterns, max_patterns=5):
        """
        Plot detected patterns on price chart.
        
        Args:
            pattern_name (str): Name of the pattern
            patterns (list): List of detected patterns
            max_patterns (int): Maximum number of patterns to plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if not patterns:
            logger.warning(f"No {pattern_name} patterns to plot")
            return None
        
        # Limit the number of patterns to plot
        patterns_to_plot = patterns[:max_patterns]
        
        fig, axes = plt.subplots(len(patterns_to_plot), 1, figsize=(12, 5 * len(patterns_to_plot)))
        
        if len(patterns_to_plot) == 1:
            axes = [axes]
        
        for i, pattern in enumerate(patterns_to_plot):
            ax = axes[i]
            
            # Get data range for this pattern
            start_idx = pattern['X_idx'] - 10
            end_idx = pattern['D_idx'] + 10
            
            if start_idx < 0:
                start_idx = 0
            if end_idx >= len(self.data):
                end_idx = len(self.data) - 1
            
            plot_data = self.data.iloc[start_idx:end_idx+1]
            
            # Plot price data
            ax.plot(plot_data.index, plot_data['close'], color='black', alpha=0.5)
            
            # Plot pattern points
            points_x = [
                self.data.index[pattern['X_idx']],
                self.data.index[pattern['A_idx']],
                self.data.index[pattern['B_idx']],
                self.data.index[pattern['C_idx']],
                self.data.index[pattern['D_idx']]
            ]
            
            points_y = [
                pattern['X_price'],
                pattern['A_price'],
                pattern['B_price'],
                pattern['C_price'],
                pattern['D_price']
            ]
            
            # Plot lines connecting pattern points
            ax.plot(points_x, points_y, 'o-', color='blue')
            
            # Add labels for pattern points
            for j, (x, y, label) in enumerate(zip(points_x, points_y, ['X', 'A', 'B', 'C', 'D'])):
                ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points')
            
            # Set title and labels
            ax.set_title(f"{pattern_name.capitalize()} Pattern {i+1} ({pattern['direction']})")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add pattern details as text
            details = (
                f"Pattern ID: {pattern['id']}\n"
                f"Direction: {pattern['direction']}\n"
                f"X Date: {pattern['X_date'].strftime('%Y-%m-%d')}\n"
                f"D Date: {pattern['D_date'].strftime('%Y-%m-%d')}\n"
                f"AB/XA Ratio: {pattern['AB_XA_ratio']:.3f}\n"
                f"BC/AB Ratio: {pattern['BC_AB_ratio']:.3f}\n"
                f"CD/BC Ratio: {pattern['CD_BC_ratio']:.3f}\n"
                f"AD/XA Ratio: {pattern['AD_XA_ratio']:.3f}"
            )
            
            ax.text(0.02, 0.98, details, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_equity_curve(self, results):
        """
        Plot equity curves for all tested patterns.
        
        Args:
            results (dict): Dictionary of pattern name to performance metrics
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for pattern_name, performance in results.items():
            if 'detailed_results' not in performance or not performance['detailed_results']:
                continue
            
            # Calculate equity curve
            equity = [self.initial_capital]
            dates = []
            
            # Sort trades by entry date
            trades = sorted(performance['detailed_results'], key=lambda x: x['entry_date'])
            
            for trade in trades:
                last_equity = equity[-1]
                trade_size = last_equity * self.position_size
                profit_amount = trade_size * trade['profit_pct']
                new_equity = last_equity + profit_amount
                
                equity.append(new_equity)
                dates.append(trade['entry_date'])
            
            # Add final date
            dates.append(self.data.index[-1])
            
            # Plot equity curve
            ax.plot(dates, equity, label=f"{pattern_name.capitalize()} ({len(trades)} trades)")
        
        # Add reference line for initial capital
        ax.axhline(y=self.initial_capital, color='black', linestyle='--', alpha=0.5)
        
        # Set title and labels
        ax.set_title('Equity Curves by Pattern')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self, results):
        """
        Generate a summary report of backtest results.
        
        Args:
            results (dict): Dictionary of pattern name to performance metrics
            
        Returns:
            pandas.DataFrame: Summary report
        """
        summary = []
        
        for pattern_name, performance in results.items():
            if 'error' in performance:
                summary.append({
                    'Pattern': pattern_name.capitalize(),
                    'Patterns Found': 0,
                    'Total Trades': 0,
                    'Win Rate': 0,
                    'Profit Factor': 0,
                    'Total Return': 0,
                    'Error': performance['error']
                })
                continue
            
            summary.append({
                'Pattern': pattern_name.capitalize(),
                'Patterns Found': performance.get('patterns_found', 0),
                'Total Trades': performance.get('total_trades', 0),
                'Win Rate': performance.get('win_rate', 0),
                'Profit Factor': performance.get('profit_factor', 0),
                'Total Return': performance.get('total_return', 0),
                'Avg Profit': performance.get('avg_profit', 0),
                'Avg Loss': performance.get('avg_loss', 0)
            })
        
        df = pd.DataFrame(summary)
        
        # Format percentage columns
        for col in ['Win Rate', 'Total Return', 'Avg Profit', 'Avg Loss']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
        
        return df 