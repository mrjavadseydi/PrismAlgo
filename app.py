#!/usr/bin/env python3
"""
Cryptocurrency Harmonic Pattern Analyzer

This application analyzes cryptocurrency data to identify which harmonic pattern
algorithm works best with a given cryptocurrency.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.config import Config
from src.data.fetcher import DataFetcher
from src.patterns.pattern_factory import PatternFactory
from src.backtest.backtest_engine import BacktestEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('harmonic_analyzer.log')
    ]
)

logger = logging.getLogger('HarmonicAnalyzer')

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Cryptocurrency Harmonic Pattern Analyzer')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    
    parser.add_argument('--api-key', type=str,
                        help='API key for data provider (overrides config file)')
    
    return parser.parse_args()

def create_output_directory(output_dir):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir (str): Path to output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

def save_results(results, backtest_engine, patterns_found, output_dir, symbol, timeframe):
    """
    Save backtest results to files.
    
    Args:
        results (dict): Backtest results
        backtest_engine (BacktestEngine): Backtest engine instance
        patterns_found (dict): Dictionary of pattern name to found patterns
        output_dir (str): Output directory
        symbol (str): Trading symbol
        timeframe (str): Timeframe
    """
    # Create timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save summary report
    summary_df = backtest_engine.generate_summary_report(results)
    summary_file = os.path.join(output_dir, f"{symbol.replace('/', '_')}_{timeframe}_summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary report to {summary_file}")
    
    # Save detailed results for each pattern
    for pattern_name, performance in results.items():
        if 'detailed_results' in performance and performance['detailed_results']:
            detailed_df = pd.DataFrame(performance['detailed_results'])
            detailed_file = os.path.join(output_dir, f"{symbol.replace('/', '_')}_{timeframe}_{pattern_name}_detailed_{timestamp}.csv")
            detailed_df.to_csv(detailed_file, index=False)
            logger.info(f"Saved detailed results for {pattern_name} to {detailed_file}")
    
    # Save equity curve plot
    equity_fig = backtest_engine.plot_equity_curve(results)
    if equity_fig:
        equity_file = os.path.join(output_dir, f"{symbol.replace('/', '_')}_{timeframe}_equity_curve_{timestamp}.png")
        equity_fig.savefig(equity_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved equity curve plot to {equity_file}")
    
    # Save pattern plots
    for pattern_name, patterns in patterns_found.items():
        if patterns:
            pattern_fig = backtest_engine.plot_patterns(pattern_name, patterns)
            if pattern_fig:
                pattern_file = os.path.join(output_dir, f"{symbol.replace('/', '_')}_{timeframe}_{pattern_name}_patterns_{timestamp}.png")
                pattern_fig.savefig(pattern_file, dpi=300, bbox_inches='tight')
                logger.info(f"Saved {pattern_name} pattern plot to {pattern_file}")

def main():
    """
    Main function to run the application.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = Config(args.config)
        
        # Create output directory
        create_output_directory(args.output_dir)
        
        # Get configuration values
        exchange = config.get('exchange')
        symbol = config.get('symbol')
        timeframe = config.get('timeframe')
        start_date = config.get('backtest.start_date')
        end_date = config.get('backtest.end_date')
        patterns = config.get_patterns()
        trading_params = config.get_trading_params()
        output_params = config.get_output_params()
        
        # Get API keys (command line argument overrides config file)
        api_keys = config.get('api_keys', [])
        if args.api_key:
            api_keys = [args.api_key]  # Use the command line key if provided
        
        logger.info(f"Analyzing {symbol} on {exchange} with timeframe {timeframe}")
        logger.info(f"Date range: {start_date} to {end_date or 'now'}")
        logger.info(f"Patterns to analyze: {', '.join(patterns)}")
        
        # Fetch data
        logger.info(f"Fetching data from {exchange}...")
        try:
            data_fetcher = DataFetcher(exchange, symbol, timeframe, api_key=api_keys)
            data = data_fetcher.fetch_ohlcv(start_date, end_date)
            
            if data.empty:
                logger.error("No data fetched. This could be due to API limits, invalid symbol, or no data available for the specified date range.")
                print("\nERROR: No data could be fetched. Possible reasons:")
                print("  - API rate limit reached (Alpha Vantage has a limit of 5 calls per minute and 500 calls per day on free tier)")
                print("  - Invalid cryptocurrency symbol")
                print("  - No data available for the specified date range")
                print("  - Invalid API key")
                print("\nPlease check the log file for more details.")
                return 1
            
            logger.info(f"Fetched {len(data)} data points from {data.index[0]} to {data.index[-1]}")
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            print(f"\nERROR: Failed to fetch data: {e}")
            print("\nIf using Alpha Vantage, please note:")
            print("  - Free tier is limited to 5 API calls per minute and 500 calls per day")
            print("  - Make sure your API key is correct")
            print("  - Try a different cryptocurrency or timeframe")
            return 1
        
        # Initialize pattern factory
        pattern_factory = PatternFactory()
        
        # Initialize backtest engine
        backtest_engine = BacktestEngine(
            data,
            initial_capital=trading_params['initial_capital'],
            position_size=trading_params['position_size'],
            stop_loss=trading_params['stop_loss'],
            take_profit=trading_params['take_profit']
        )
        
        # Run backtest for each pattern
        logger.info("Running backtests...")
        results = backtest_engine.run_backtest(patterns)
        
        # Find patterns for visualization
        patterns_found = {}
        for pattern_name in patterns:
            pattern = pattern_factory.get_pattern(pattern_name)
            patterns_found[pattern_name] = pattern.find_patterns(data)
        
        # Display results
        summary_df = backtest_engine.generate_summary_report(results)
        print("\nBacktest Results Summary:")
        print(summary_df.to_string(index=False))
        
        # Find best performing pattern
        if summary_df.empty:
            logger.warning("No patterns found in the data")
        else:
            # Convert percentage strings back to floats for comparison
            summary_df['Total Return'] = summary_df['Total Return'].apply(
                lambda x: float(x.strip('%')) / 100 if isinstance(x, str) else x
            )
            
            best_pattern = summary_df.loc[summary_df['Total Return'].idxmax()]
            
            print(f"\nBest performing pattern: {best_pattern['Pattern']}")
            print(f"Total Return: {best_pattern['Total Return']:.2%}")
            print(f"Win Rate: {best_pattern['Win Rate']}")
            print(f"Profit Factor: {best_pattern['Profit Factor']}")
        
        # Save results if enabled
        if output_params['save_results']:
            save_results(results, backtest_engine, patterns_found, args.output_dir, symbol, timeframe)
        
        # Show plots if enabled
        if output_params['plot_charts']:
            plt.show()
        
        logger.info("Analysis completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
