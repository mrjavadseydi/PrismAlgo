import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import logging
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

logger = logging.getLogger('plotter')

def plot_patterns(data, patterns, figsize=(12, 8)):
    """
    Plot detected harmonic patterns on price chart.
    
    Args:
        data (pandas.DataFrame): OHLCV data
        patterns (list): List of detected patterns
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    if not patterns:
        logger.warning("No patterns to plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(data.index, data['close'], label='Close Price')
        ax.set_title('Price Chart (No Patterns Detected)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot price
    ax.plot(data.index, data['close'], label='Close Price', color='black', alpha=0.5)
    
    # Define colors for different pattern types
    pattern_colors = {
        'butterfly': 'blue',
        'gartley': 'green',
        'bat': 'purple',
        'crab': 'orange',
        'shark': 'red',
        'cypher': 'brown'
    }
    
    # Plot each pattern
    for pattern in patterns:
        pattern_type = pattern['pattern']
        direction = pattern['direction']
        points = pattern['points']
        
        # Get color for pattern type
        color = pattern_colors.get(pattern_type.lower(), 'gray')
        
        # Adjust color based on direction
        if direction == 'bearish':
            color = f'dark{color}'
        
        # Extract x and y coordinates
        x_dates = [point[0] for point in points]
        y_prices = [point[1] for point in points]
        
        # Plot pattern lines
        ax.plot(x_dates, y_prices, marker='o', linestyle='-', color=color, 
                label=f"{pattern_type} ({direction})", alpha=0.7)
        
        # Add pattern labels
        for i, (date, price) in enumerate(points):
            ax.annotate(f"{chr(65 + i)}", (date, price), 
                       xytext=(5, 5), textcoords='offset points')
        
        # Highlight potential reversal zone
        if 'potential_reversal' in pattern and pattern['potential_reversal']:
            reversal_date = points[-1][0]
            reversal_price = points[-1][1]
            
            # Draw a vertical line at the reversal point
            ax.axvline(x=reversal_date, color=color, linestyle='--', alpha=0.5)
            
            # Add a marker for the reversal point
            ax.scatter([reversal_date], [reversal_price], color=color, s=100, 
                      marker='*', label=f"{pattern_type} Reversal")
    
    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    
    # Add labels and title
    ax.set_title('Harmonic Patterns Detection')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    
    # Add legend (only show one entry per pattern type)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    ax.legend(unique_handles, unique_labels, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    logger.info(f"Created chart with {len(patterns)} patterns")
    return fig

def plot_performance(performance_results, figsize=(10, 12)):
    """
    Plot performance metrics for detected patterns.
    
    Args:
        performance_results (dict): Performance metrics
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Extract pattern types and metrics
    pattern_types = list(performance_results['by_pattern'].keys())
    win_rates = [performance_results['by_pattern'][p]['win_rate'] for p in pattern_types]
    profit_factors = [performance_results['by_pattern'][p]['profit_factor'] for p in pattern_types]
    total_returns = [performance_results['by_pattern'][p]['total_return'] for p in pattern_types]
    
    # Add overall results
    pattern_types.append('Overall')
    win_rates.append(performance_results['overall']['win_rate'])
    profit_factors.append(performance_results['overall']['profit_factor'])
    total_returns.append(performance_results['overall']['total_return'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Plot win rates
    axes[0].bar(pattern_types, win_rates, color='skyblue')
    axes[0].set_title('Win Rate by Pattern')
    axes[0].set_ylabel('Win Rate')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)  # 50% reference line
    
    # Add value labels
    for i, v in enumerate(win_rates):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    # Plot profit factors (cap at 10 for visualization)
    capped_profit_factors = [min(pf, 10) for pf in profit_factors]
    bars = axes[1].bar(pattern_types, capped_profit_factors, color='lightgreen')
    axes[1].set_title('Profit Factor by Pattern')
    axes[1].set_ylabel('Profit Factor')
    axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.5)  # Breakeven reference line
    
    # Add value labels
    for i, (v, original) in enumerate(zip(capped_profit_factors, profit_factors)):
        if original > 10:
            axes[1].text(i, v - 0.5, f"{original:.1f}", ha='center')
        else:
            axes[1].text(i, v + 0.2, f"{original:.2f}", ha='center')
    
    # Plot total returns
    colors = ['green' if r >= 0 else 'red' for r in total_returns]
    axes[2].bar(pattern_types, total_returns, color=colors)
    axes[2].set_title('Total Return by Pattern')
    axes[2].set_ylabel('Total Return (%)')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)  # Zero reference line
    
    # Add value labels
    for i, v in enumerate(total_returns):
        if v >= 0:
            axes[2].text(i, v + 1, f"{v:.1f}%", ha='center')
        else:
            axes[2].text(i, v - 2, f"{v:.1f}%", ha='center')
    
    # Adjust layout
    plt.tight_layout()
    
    logger.info("Created performance chart")
    return fig

def plot_price_action(data, price_action_results, figsize=(12, 15)):
    """
    Plot price action analysis results.
    
    Args:
        data (pandas.DataFrame): OHLCV data
        price_action_results (dict): Price action analysis results
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot price with support/resistance levels
    axes[0].plot(data.index, data['close'], label='Close Price')
    
    # Add support and resistance levels
    if 'market_structure' in price_action_results:
        for level in price_action_results['market_structure']['support_levels']:
            axes[0].axhline(y=level, color='g', linestyle='--', alpha=0.5)
        for level in price_action_results['market_structure']['resistance_levels']:
            axes[0].axhline(y=level, color='r', linestyle='--', alpha=0.5)
    
    # Add buy/sell signals if available
    if 'pa_buy_signal' in data.columns and 'pa_sell_signal' in data.columns:
        axes[0].scatter(data.index[data['pa_buy_signal']], 
                       data.loc[data['pa_buy_signal'], 'low'] * 0.99, 
                       marker='^', color='g', s=100)
        axes[0].scatter(data.index[data['pa_sell_signal']], 
                       data.loc[data['pa_sell_signal'], 'high'] * 1.01, 
                       marker='v', color='r', s=100)
    
    axes[0].set_title('Price with Support/Resistance Levels')
    axes[0].legend()
    
    # Plot volume
    axes[1].bar(data.index, data['volume'], color='blue', alpha=0.5)
    if 'volume' in price_action_results:
        # Highlight volume spikes
        volume_spikes = price_action_results['volume']['volume_spikes']
        axes[1].bar(data.index[volume_spikes], 
                   data.loc[volume_spikes, 'volume'], 
                   color='red', alpha=0.7)
    
    axes[1].set_title('Volume Analysis')
    
    # Plot candlestick patterns (as markers on a separate subplot)
    if 'candlestick' in price_action_results:
        patterns = price_action_results['candlestick']['patterns']
        
        # Create a dummy line for the third subplot
        axes[2].plot(data.index, data['close'], alpha=0)
        
        # Add markers for different patterns
        pattern_colors = {
            'doji': 'blue',
            'hammer': 'green',
            'shooting_star': 'red',
            'pin_bar': 'purple',
            'bullish_engulfing': 'lime',
            'bearish_engulfing': 'orange',
            'morning_star': 'cyan',
            'evening_star': 'magenta'
        }
        
        for pattern_name, pattern_series in patterns.items():
            if pattern_name in pattern_colors and pattern_series.any():
                # Plot markers at the bottom of the chart
                y_pos = axes[2].get_ylim()[0] + (axes[2].get_ylim()[1] - axes[2].get_ylim()[0]) * 0.05 * (list(pattern_colors.keys()).index(pattern_name) + 1)
                axes[2].scatter(data.index[pattern_series], 
                              [y_pos] * pattern_series.sum(), 
                              marker='o', 
                              color=pattern_colors[pattern_name], 
                              label=pattern_name)
    
    axes[2].set_title('Candlestick Patterns')
    axes[2].legend(loc='upper left')
    
    # Format x-axis to show dates nicely
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[0].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    logger.info("Created price action chart")
    return fig 