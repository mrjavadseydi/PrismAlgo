import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HarmonicPattern')

class BaseHarmonicPattern(ABC):
    """
    Base class for all harmonic pattern implementations.
    """
    
    def __init__(self, name, tolerance=0.05):
        """
        Initialize the harmonic pattern detector.
        
        Args:
            name (str): Name of the pattern
            tolerance (float): Tolerance for Fibonacci ratio matching (default: 0.05)
        """
        self.name = name
        self.tolerance = tolerance
        self.patterns_found = []
        logger.info(f"Initialized {name} pattern detector with tolerance {tolerance}")
    
    @abstractmethod
    def get_pattern_ratios(self):
        """
        Get the ideal Fibonacci ratios for this pattern.
        
        Returns:
            dict: Dictionary of ratios for each leg of the pattern
        """
        pass
    
    def is_ratio_valid(self, actual_ratio, target_ratio):
        """
        Check if the actual ratio is within tolerance of the target ratio.
        
        Args:
            actual_ratio (float): Actual ratio calculated from price data
            target_ratio (float): Target Fibonacci ratio for the pattern
            
        Returns:
            bool: True if the ratio is valid, False otherwise
        """
        return abs(actual_ratio - target_ratio) <= self.tolerance
    
    def find_swings(self, df, window=5):
        """
        Find swing highs and lows in the price data.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            window (int): Window size for detecting swings
            
        Returns:
            tuple: (swing_highs, swing_lows) DataFrames with swing points
        """
        highs = df['high'].values
        lows = df['low'].values
        
        # Find swing highs
        swing_highs = []
        for i in range(window, len(highs) - window):
            if all(highs[i] > highs[i-j] for j in range(1, window+1)) and \
               all(highs[i] > highs[i+j] for j in range(1, window+1)):
                swing_highs.append((df.index[i], highs[i], i))
        
        # Find swing lows
        swing_lows = []
        for i in range(window, len(lows) - window):
            if all(lows[i] < lows[i-j] for j in range(1, window+1)) and \
               all(lows[i] < lows[i+j] for j in range(1, window+1)):
                swing_lows.append((df.index[i], lows[i], i))
        
        # Convert to DataFrames
        swing_highs_df = pd.DataFrame(swing_highs, columns=['timestamp', 'price', 'index'])
        swing_lows_df = pd.DataFrame(swing_lows, columns=['timestamp', 'price', 'index'])
        
        logger.info(f"Found {len(swing_highs_df)} swing highs and {len(swing_lows_df)} swing lows")
        return swing_highs_df, swing_lows_df
    
    def calculate_ratio(self, start_price, end_price, reference_price):
        """
        Calculate the retracement ratio between price points.
        
        Args:
            start_price (float): Starting price point
            end_price (float): Ending price point
            reference_price (float): Reference price point
            
        Returns:
            float: Retracement ratio
        """
        if reference_price == start_price:
            return 0.0
        
        return abs((end_price - start_price) / (reference_price - start_price))
    
    @abstractmethod
    def find_patterns(self, df, swing_window=5):
        """
        Find the harmonic pattern in the given price data.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            swing_window (int): Window size for detecting swings
            
        Returns:
            list: List of detected patterns with their details
        """
        pass
    
    def evaluate_performance(self, df, patterns, stop_loss=0.02, take_profit=0.05):
        """
        Evaluate the performance of detected patterns.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            patterns (list): List of detected patterns
            stop_loss (float): Stop loss percentage
            take_profit (float): Take profit percentage
            
        Returns:
            dict: Performance metrics
        """
        if not patterns:
            logger.warning("No patterns to evaluate")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_return': 0
            }
        
        results = []
        
        for pattern in patterns:
            entry_index = pattern['D_index']
            if entry_index >= len(df) - 1:
                continue  # Skip if pattern is at the end of the data
                
            entry_price = df.iloc[entry_index]['close']
            stop_price = entry_price * (1 - stop_loss) if pattern['direction'] == 'bullish' else entry_price * (1 + stop_loss)
            target_price = entry_price * (1 + take_profit) if pattern['direction'] == 'bullish' else entry_price * (1 - take_profit)
            
            # Check future price action
            hit_target = False
            hit_stop = False
            exit_price = None
            exit_index = None
            
            for i in range(entry_index + 1, len(df)):
                current_high = df.iloc[i]['high']
                current_low = df.iloc[i]['low']
                
                if pattern['direction'] == 'bullish':
                    if current_high >= target_price:
                        hit_target = True
                        exit_price = target_price
                        exit_index = i
                        break
                    if current_low <= stop_price:
                        hit_stop = True
                        exit_price = stop_price
                        exit_index = i
                        break
                else:  # bearish
                    if current_low <= target_price:
                        hit_target = True
                        exit_price = target_price
                        exit_index = i
                        break
                    if current_high >= stop_price:
                        hit_stop = True
                        exit_price = stop_price
                        exit_index = i
                        break
            
            # If neither target nor stop was hit, use the last price
            if not hit_target and not hit_stop and i == len(df) - 1:
                exit_price = df.iloc[-1]['close']
                exit_index = len(df) - 1
            
            if exit_price is not None:
                profit_pct = (exit_price - entry_price) / entry_price if pattern['direction'] == 'bullish' else (entry_price - exit_price) / entry_price
                
                results.append({
                    'pattern_id': pattern['id'],
                    'entry_date': df.index[entry_index],
                    'exit_date': df.index[exit_index] if exit_index is not None else None,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': pattern['direction'],
                    'profit_pct': profit_pct,
                    'hit_target': hit_target,
                    'hit_stop': hit_stop,
                    'bars_held': exit_index - entry_index if exit_index is not None else None
                })
        
        # Calculate performance metrics
        if not results:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_return': 0
            }
        
        total_trades = len(results)
        winning_trades = sum(1 for r in results if r['profit_pct'] > 0)
        losing_trades = sum(1 for r in results if r['profit_pct'] <= 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        profits = [r['profit_pct'] for r in results if r['profit_pct'] > 0]
        losses = [abs(r['profit_pct']) for r in results if r['profit_pct'] <= 0]
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        total_profit = sum(profits)
        total_loss = sum(losses)
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        total_return = sum(r['profit_pct'] for r in results)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'detailed_results': results
        } 