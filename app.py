#!/usr/bin/env python3
"""
Cryptocurrency Harmonic Pattern Analyzer

This application analyzes cryptocurrency data to identify which harmonic pattern
algorithm works best with a given cryptocurrency.
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
import traceback
import json
from datetime import datetime
import matplotlib.pyplot as plt

from src.patterns.harmonic_analyzer import HarmonicPatternAnalyzer
from src.visualization.plotter import plot_patterns
from src.price_action.price_action_analyzer import PriceActionAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('app')

def run_backtest(data, config):
    """Run backtest with the given data and configuration."""
    try:
        # Initialize analyzers
        harmonic_analyzer = HarmonicPatternAnalyzer(data, config)
        price_action_analyzer = PriceActionAnalyzer(data, config.get('price_action', {}))
        
        # Run analysis
        harmonic_patterns = harmonic_analyzer.analyze()
        price_action_results = price_action_analyzer.analyze()
        price_action_summary = price_action_analyzer.get_analysis_summary()
        
        # Get signals
        harmonic_signals = harmonic_analyzer.get_pattern_signals()
        price_action_signals = price_action_analyzer.get_combined_signals()
        
        # Run backtest for harmonic patterns
        backtest_results = {}
        for pattern_name in harmonic_patterns:
            pattern_signals = harmonic_signals[harmonic_signals['pattern'] == pattern_name]
            if not pattern_signals.empty:
                backtest_results[pattern_name] = backtest_pattern(data, pattern_signals)
        
        # Convert price action signals to the format expected by backtest_pattern
        if not price_action_signals.empty:
            # Create a new DataFrame with the expected format
            pa_signals_converted = pd.DataFrame(index=price_action_signals.index)
            pa_signals_converted['signal'] = None
            
            # Set 'buy' signals
            buy_indices = price_action_signals[price_action_signals['buy']].index
            if not buy_indices.empty:
                pa_signals_converted.loc[buy_indices, 'signal'] = 'buy'
            
            # Set 'sell' signals
            sell_indices = price_action_signals[price_action_signals['sell']].index
            if not sell_indices.empty:
                pa_signals_converted.loc[sell_indices, 'signal'] = 'sell'
            
            # Remove rows with no signal
            pa_signals_converted = pa_signals_converted.dropna()
            
            # Run backtest for price action signals
            if not pa_signals_converted.empty:
                backtest_results['Price Action'] = backtest_pattern(data, pa_signals_converted)
        
        # Create summary DataFrame
        summary_data = []
        for pattern, results in backtest_results.items():
            summary_data.append({
                'Pattern': pattern,
                'Total Trades': results['total_trades'],
                'Win Rate': f"{results['win_rate']:.2%}",
                'Profit Factor': f"{results['profit_factor']:.2f}",
                'Total Return': f"{results['total_return']:.2%}",
                'Max Drawdown': f"{results['max_drawdown']:.2%}"
            })
        
        # Add price action results to summary if not already included in backtest_results
        if price_action_summary and 'Price Action' not in backtest_results:
            price_action_row = {
                'Pattern': 'Price Action',
                'Total Trades': price_action_summary['buy_signals'] + price_action_summary['sell_signals'],
                'Win Rate': f"{price_action_summary['win_rate']:.2%}",
                'Profit Factor': f"{price_action_summary['profit_factor']:.2f}",
                'Total Return': f"{price_action_summary['total_return']:.2%}",
                'Max Drawdown': f"{price_action_summary['max_drawdown']:.2%}"
            }
            summary_data.append(price_action_row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Print summary
        print("\n=== Backtest Results Summary ===")
        if not summary_df.empty:
            print(summary_df.to_string(index=False))
        else:
            print("No patterns detected for backtesting.")
        
        # Plot patterns
        try:
            pattern_list = []
            for pattern_name, patterns in harmonic_patterns.items():
                logger.debug(f"Processing {pattern_name} patterns: {type(patterns)}")
                if isinstance(patterns, pd.DataFrame):
                    # Process DataFrame
                    for idx, pattern in patterns.iterrows():
                        logger.debug(f"Pattern from DataFrame: {pattern}")
                        if isinstance(pattern, dict):
                            pattern_dict = pattern
                        else:
                            pattern_dict = {
                                'direction': pattern.get('direction', 'bullish'),
                                'points': pattern.get('points', {}),
                                'potential_reversal': pattern.get('potential_reversal', False),
                                'completion_percentage': pattern.get('completion_percentage', 0)
                            }
                        pattern_dict['pattern_name'] = pattern_name
                        pattern_list.append(pattern_dict)
                else:
                    # Process list
                    for i, pattern in enumerate(patterns):
                        logger.debug(f"Pattern from list [{i}]: {pattern}")
                        if isinstance(pattern, dict):
                            pattern_dict = pattern.copy()  # Create a copy to avoid modifying the original
                            if 'pattern_name' not in pattern_dict:
                                pattern_dict['pattern_name'] = pattern_name
                            pattern_list.append(pattern_dict)
            
            if pattern_list:
                logger.debug(f"Pattern list for plotting: {pattern_list}")
                plot_patterns(data, pattern_list)
        except Exception as e:
            logger.error(f"Error plotting patterns: {e}")
            logger.debug(f"Pattern list that caused error: {pattern_list}")
        
        return {
            'harmonic_patterns': harmonic_patterns,
            'price_action_results': price_action_results,
            'backtest_results': backtest_results,
            'summary': summary_df
        }
    
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        traceback.print_exc()
        return None

def fetch_data(symbol, interval, limit):
    """
    Fetch OHLCV data for the given symbol and interval.
    
    Args:
        symbol (str): Symbol to fetch data for
        interval (str): Timeframe interval
        limit (int): Number of candles to fetch
        
    Returns:
        pandas.DataFrame: DataFrame with OHLCV data
    """
    try:
        # Import ccxt only when needed to avoid dependency issues
        import ccxt
        
        # Initialize exchange
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, interval, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        # If ccxt fails, try using a mock dataset for testing
        logger.info("Using mock data for testing")
        return generate_mock_data(limit)

def generate_mock_data(n_periods=500):
    """
    Generate mock OHLCV data for testing.
    
    Args:
        n_periods (int): Number of periods to generate
        
    Returns:
        pandas.DataFrame: DataFrame with mock OHLCV data
    """
    # Generate dates
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=n_periods)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_periods)
    
    # Generate price data with some randomness and trend
    np.random.seed(42)  # For reproducibility
    
    # Start with a base price
    base_price = 10000
    
    # Generate random walk
    random_walk = np.random.normal(0, 1, n_periods).cumsum()
    
    # Add some trend
    trend = np.linspace(0, 2000, n_periods)
    
    # Add some cyclicality
    cycles = 1000 * np.sin(np.linspace(0, 4*np.pi, n_periods))
    
    # Combine components
    close_prices = base_price + random_walk + trend + cycles
    
    # Generate OHLC based on close prices
    daily_volatility = 0.02
    high_prices = close_prices * (1 + np.random.uniform(0, daily_volatility, n_periods))
    low_prices = close_prices * (1 - np.random.uniform(0, daily_volatility, n_periods))
    open_prices = low_prices + np.random.uniform(0, 1, n_periods) * (high_prices - low_prices)
    
    # Generate volume
    volume = np.random.uniform(100, 1000, n_periods) * (1 + 0.5 * np.abs(np.random.normal(0, 1, n_periods)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return df

def generate_report(symbol, interval, results):
    """
    Generate a report from the backtest results.
    
    Args:
        symbol (str): Symbol that was analyzed
        interval (str): Timeframe interval
        results (dict): Results from the backtest
        
    Returns:
        str: Path to the generated report
    """
    try:
        # Create output directory
        output_dir = 'reports'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Create report filename
        report_file = os.path.join(output_dir, f"{symbol}_{interval}_{timestamp}_report.html")
        
        # Extract data from results
        harmonic_patterns = results.get('harmonic_patterns', {})
        price_action_results = results.get('price_action_results', {})
        backtest_results = results.get('backtest_results', {})
        summary_df = results.get('summary', pd.DataFrame())
        
        # Generate HTML report
        with open(report_file, 'w') as f:
            f.write(f"""
            <html>
            <head>
                <title>Harmonic Pattern Analysis Report - {symbol} {interval}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333366; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .section {{ margin-bottom: 30px; }}
                    .highlight {{ background-color: #ffffcc; }}
                </style>
            </head>
            <body>
                <h1>Harmonic Pattern Analysis Report</h1>
                <div class="section">
                    <h2>Analysis Information</h2>
                    <table>
                        <tr><th>Symbol</th><td>{symbol}</td></tr>
                        <tr><th>Interval</th><td>{interval}</td></tr>
                        <tr><th>Date Generated</th><td>{timestamp}</td></tr>
                    </table>
                </div>
            """)
            
            # Add summary table
            if not summary_df.empty:
                f.write(f"""
                <div class="section">
                    <h2>Backtest Results Summary</h2>
                    <table>
                        <tr>
                            {"".join(f"<th>{col}</th>" for col in summary_df.columns)}
                        </tr>
                        {"".join(f"<tr>{''.join(f'<td>{cell}</td>' for cell in row)}</tr>" for _, row in summary_df.iterrows())}
                    </table>
                </div>
                """)
            
            # Add harmonic patterns section
            if harmonic_patterns:
                f.write(f"""
                <div class="section">
                    <h2>Harmonic Patterns Detected</h2>
                    <table>
                        <tr>
                            <th>Pattern</th>
                            <th>Count</th>
                        </tr>
                """)
                
                for pattern_name, patterns in harmonic_patterns.items():
                    pattern_count = len(patterns) if hasattr(patterns, '__len__') else 0
                    f.write(f"<tr><td>{pattern_name}</td><td>{pattern_count}</td></tr>")
                
                f.write("</table></div>")
            
            # Add price action section
            if price_action_results:
                f.write("<div class='section'><h2>Price Action Analysis</h2>")
                
                # Market structure
                if 'market_structure' in price_action_results:
                    ms = price_action_results['market_structure']
                    f.write(f"""
                    <h3>Market Structure</h3>
                    <table>
                        <tr><th>Trend</th><td>{ms.get('trend', 'Unknown')}</td></tr>
                        <tr><th>Support Levels</th><td>{len(ms.get('support_levels', []))}</td></tr>
                        <tr><th>Resistance Levels</th><td>{len(ms.get('resistance_levels', []))}</td></tr>
                    </table>
                    """)
                
                # Candlestick patterns
                if 'candlestick' in price_action_results:
                    cs = price_action_results['candlestick']
                    f.write("<h3>Candlestick Patterns</h3><table><tr><th>Pattern</th><th>Count</th></tr>")
                    
                    for pattern_name, pattern_series in cs.get('patterns', {}).items():
                        if hasattr(pattern_series, 'sum'):
                            pattern_count = pattern_series.sum()
                            f.write(f"<tr><td>{pattern_name}</td><td>{pattern_count}</td></tr>")
                    
                    f.write("</table>")
                
                # Advanced patterns
                if 'pattern_recognition' in price_action_results:
                    pr = price_action_results['pattern_recognition']
                    f.write("<h3>Advanced Patterns</h3><table>")
                    
                    # Channels
                    channels = pr.get('channels', [])
                    f.write(f"<tr><th>Channels</th><td>{len(channels)}</td></tr>")
                    
                    # Wedges
                    wedges = pr.get('wedges', [])
                    f.write(f"<tr><th>Wedges</th><td>{len(wedges)}</td></tr>")
                    
                    # Double patterns
                    double_patterns = pr.get('double_patterns', {})
                    double_tops = double_patterns.get('double_top', [])
                    double_bottoms = double_patterns.get('double_bottom', [])
                    f.write(f"<tr><th>Double Tops</th><td>{len(double_tops)}</td></tr>")
                    f.write(f"<tr><th>Double Bottoms</th><td>{len(double_bottoms)}</td></tr>")
                    
                    # Zigzag patterns
                    zigzags = pr.get('zigzag_patterns', [])
                    f.write(f"<tr><th>Zigzag Patterns</th><td>{len(zigzags)}</td></tr>")
                    
                    f.write("</table>")
                
                f.write("</div>")
            
            # Close HTML
            f.write("</body></html>")
        
        logger.info(f"Report generated: {report_file}")
        return report_file
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return None

def backtest_pattern(data, signals):
    """
    Backtest a pattern based on signals.
    
    Args:
        data (pandas.DataFrame): OHLCV data
        signals (pandas.DataFrame): DataFrame with buy and sell signals
        
    Returns:
        dict: Backtest results
    """
    try:
        # Initialize results
        results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 1.0,
            'total_return': 0.0,
            'max_drawdown': 0.0
        }
        
        # If no signals, return default results
        if signals.empty:
            return results
        
        # Create a copy of the data to avoid modifying the original
        backtest_data = data.copy()
        
        # Add signals to the data
        backtest_data['buy_signal'] = False
        backtest_data['sell_signal'] = False
        
        # Map signals to the backtest data
        for idx, row in signals.iterrows():
            if idx in backtest_data.index:
                if row.get('signal') == 'buy':
                    backtest_data.loc[idx, 'buy_signal'] = True
                elif row.get('signal') == 'sell':
                    backtest_data.loc[idx, 'sell_signal'] = True
        
        # Initialize position and equity columns
        backtest_data['position'] = 0
        backtest_data['equity'] = 0.0
        
        # Set initial position based on first signal
        in_position = False
        entry_price = 0.0
        position_type = None  # 'long' or 'short'
        trades = []
        
        # Simulate trades
        for i in range(1, len(backtest_data)):
            # Check for buy signal
            if backtest_data['buy_signal'].iloc[i] and not in_position:
                in_position = True
                position_type = 'long'
                entry_price = backtest_data['close'].iloc[i]
                backtest_data.loc[backtest_data.index[i], 'position'] = 1
            
            # Check for sell signal
            elif backtest_data['sell_signal'].iloc[i] and not in_position:
                in_position = True
                position_type = 'short'
                entry_price = backtest_data['close'].iloc[i]
                backtest_data.loc[backtest_data.index[i], 'position'] = -1
            
            # Check for exit (opposite signal)
            elif (backtest_data['sell_signal'].iloc[i] and in_position and position_type == 'long') or \
                 (backtest_data['buy_signal'].iloc[i] and in_position and position_type == 'short'):
                exit_price = backtest_data['close'].iloc[i]
                
                # Calculate profit/loss
                if position_type == 'long':
                    profit_pct = (exit_price - entry_price) / entry_price
                else:  # short
                    profit_pct = (entry_price - exit_price) / entry_price
                
                # Record trade
                trades.append({
                    'entry_date': backtest_data.index[i-1],
                    'exit_date': backtest_data.index[i],
                    'position_type': position_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct
                })
                
                # Reset position
                in_position = False
                entry_price = 0.0
                position_type = None
                backtest_data.loc[backtest_data.index[i], 'position'] = 0
        
        # Calculate equity curve
        initial_equity = 10000  # Arbitrary starting capital
        current_equity = initial_equity
        equity_curve = [initial_equity]
        
        for trade in trades:
            trade_profit = current_equity * trade['profit_pct']
            current_equity += trade_profit
            equity_curve.append(current_equity)
        
        # Calculate metrics
        if trades:
            # Total trades
            results['total_trades'] = len(trades)
            
            # Win rate
            winning_trades = sum(1 for trade in trades if trade['profit_pct'] > 0)
            results['winning_trades'] = winning_trades
            results['losing_trades'] = len(trades) - winning_trades
            results['win_rate'] = winning_trades / len(trades)
            
            # Profit factor
            gross_profit = sum(trade['profit_pct'] for trade in trades if trade['profit_pct'] > 0)
            gross_loss = abs(sum(trade['profit_pct'] for trade in trades if trade['profit_pct'] < 0))
            results['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Total return
            results['total_return'] = (current_equity - initial_equity) / initial_equity
            
            # Max drawdown
            peak = initial_equity
            drawdown = 0
            max_drawdown = 0
            
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            results['max_drawdown'] = max_drawdown
        
        return results
    
    except Exception as e:
        logger.error(f"Error in backtest_pattern: {e}")
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 1.0,
            'total_return': 0.0,
            'max_drawdown': 0.0
        }

def main():
    """
    Main function to run the application.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Harmonic Pattern Analyzer')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol to analyze')
    parser.add_argument('--interval', type=str, default='1d', help='Timeframe interval')
    parser.add_argument('--limit', type=int, default=500, help='Number of candles to fetch')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set up logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        config = {}
    
    # Fetch data
    try:
        data = fetch_data(args.symbol, args.interval, args.limit)
        logger.info(f"Fetched {len(data)} candles for {args.symbol} on {args.interval} timeframe")
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return
    
    # Run backtest
    results = run_backtest(data, config)
    
    if results:
        # Generate report
        try:
            report = generate_report(args.symbol, args.interval, results)
            logger.info(f"Generated report: {report}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")

if __name__ == "__main__":
    main()
