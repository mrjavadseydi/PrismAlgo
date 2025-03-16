import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('performance')

def calculate_performance(data, patterns, trading_params=None):
    """
    Calculate performance metrics for detected patterns.
    
    Args:
        data (pandas.DataFrame): OHLCV data
        patterns (list): List of detected patterns
        trading_params (dict): Trading parameters
        
    Returns:
        dict: Performance metrics
    """
    if not patterns:
        logger.warning("No patterns provided for performance calculation")
        return {
            'overall': {
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'trades': 0
            },
            'by_pattern': {}
        }
    
    # Default trading parameters
    default_params = {
        'risk_per_trade': 0.02,
        'stop_loss_atr_multiplier': 1.5,
        'take_profit_atr_multiplier': 3.0,
        'trailing_stop': False,
        'trailing_stop_activation': 0.01,
        'trailing_stop_step': 0.005
    }
    
    # Merge with provided parameters
    params = {**default_params, **(trading_params or {})}
    
    # Calculate ATR for stop loss and take profit
    data['atr'] = calculate_atr(data, 14)
    
    # Initialize results
    results = {
        'overall': {
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'trades': 0
        },
        'by_pattern': {}
    }
    
    # Initialize pattern-specific results
    pattern_types = set(p['pattern'] for p in patterns)
    for pattern_type in pattern_types:
        results['by_pattern'][pattern_type] = {
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'trades': 0
        }
    
    # Process each pattern
    for pattern in patterns:
        pattern_type = pattern['pattern']
        direction = pattern['direction']
        
        # Get pattern completion point (last point)
        completion_idx = data.index.get_loc(pattern['points'][-1][0])
        
        # Skip if not enough data after pattern completion
        if completion_idx >= len(data) - 1:
            continue
        
        # Entry price is the close at pattern completion
        entry_price = data['close'].iloc[completion_idx]
        
        # ATR at entry for stop loss and take profit calculation
        atr_at_entry = data['atr'].iloc[completion_idx]
        
        # Calculate stop loss and take profit levels
        if direction == 'bullish':
            stop_loss = entry_price - (atr_at_entry * params['stop_loss_atr_multiplier'])
            take_profit = entry_price + (atr_at_entry * params['take_profit_atr_multiplier'])
        else:  # bearish
            stop_loss = entry_price + (atr_at_entry * params['stop_loss_atr_multiplier'])
            take_profit = entry_price - (atr_at_entry * params['take_profit_atr_multiplier'])
        
        # Simulate trade
        trade_result = simulate_trade(
            data.iloc[completion_idx+1:], 
            direction, 
            entry_price, 
            stop_loss, 
            take_profit,
            params['trailing_stop'],
            params['trailing_stop_activation'],
            params['trailing_stop_step']
        )
        
        # Update results
        if trade_result['result'] == 'win':
            results['overall']['wins'] += 1
            results['by_pattern'][pattern_type]['wins'] += 1
        else:
            results['overall']['losses'] += 1
            results['by_pattern'][pattern_type]['losses'] += 1
        
        results['overall']['total_return'] += trade_result['return_pct']
        results['by_pattern'][pattern_type]['total_return'] += trade_result['return_pct']
        
        results['overall']['trades'] += 1
        results['by_pattern'][pattern_type]['trades'] += 1
    
    # Calculate metrics for overall results
    if results['overall']['trades'] > 0:
        results['overall']['win_rate'] = results['overall']['wins'] / results['overall']['trades']
        
        # Avoid division by zero
        if results['overall']['losses'] > 0:
            results['overall']['profit_factor'] = (
                results['overall']['wins'] * params['take_profit_atr_multiplier'] / 
                (results['overall']['losses'] * params['stop_loss_atr_multiplier'])
            )
        else:
            results['overall']['profit_factor'] = float('inf')
    
    # Calculate metrics for each pattern type
    for pattern_type in results['by_pattern']:
        if results['by_pattern'][pattern_type]['trades'] > 0:
            results['by_pattern'][pattern_type]['win_rate'] = (
                results['by_pattern'][pattern_type]['wins'] / 
                results['by_pattern'][pattern_type]['trades']
            )
            
            # Avoid division by zero
            if results['by_pattern'][pattern_type]['losses'] > 0:
                results['by_pattern'][pattern_type]['profit_factor'] = (
                    results['by_pattern'][pattern_type]['wins'] * params['take_profit_atr_multiplier'] / 
                    (results['by_pattern'][pattern_type]['losses'] * params['stop_loss_atr_multiplier'])
                )
            else:
                results['by_pattern'][pattern_type]['profit_factor'] = float('inf')
    
    logger.info(f"Calculated performance for {results['overall']['trades']} trades")
    return results

def calculate_atr(data, period=14):
    """
    Calculate Average True Range (ATR).
    
    Args:
        data (pandas.DataFrame): OHLCV data
        period (int): ATR period
        
    Returns:
        pandas.Series: ATR values
    """
    high = data['high']
    low = data['low']
    close = data['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr

def simulate_trade(data, direction, entry_price, stop_loss, take_profit, 
                  use_trailing_stop=False, trailing_activation=0.01, trailing_step=0.005):
    """
    Simulate a trade and calculate its outcome.
    
    Args:
        data (pandas.DataFrame): OHLCV data after entry
        direction (str): 'bullish' or 'bearish'
        entry_price (float): Entry price
        stop_loss (float): Stop loss price
        take_profit (float): Take profit price
        use_trailing_stop (bool): Whether to use trailing stop
        trailing_activation (float): Percentage of profit to activate trailing stop
        trailing_step (float): Trailing stop step size
        
    Returns:
        dict: Trade result
    """
    result = {
        'result': None,
        'exit_price': None,
        'exit_date': None,
        'return_pct': 0,
        'bars_held': 0
    }
    
    # Initialize trailing stop variables
    trailing_stop_active = False
    trailing_stop_level = stop_loss
    
    # Simulate price movement
    for i, row in data.iterrows():
        result['bars_held'] += 1
        
        if direction == 'bullish':
            # Check if stop loss hit
            if row['low'] <= stop_loss:
                result['result'] = 'loss'
                result['exit_price'] = stop_loss
                result['exit_date'] = i
                result['return_pct'] = (stop_loss - entry_price) / entry_price * 100
                break
            
            # Check if take profit hit
            if row['high'] >= take_profit:
                result['result'] = 'win'
                result['exit_price'] = take_profit
                result['exit_date'] = i
                result['return_pct'] = (take_profit - entry_price) / entry_price * 100
                break
            
            # Check trailing stop if active
            if use_trailing_stop:
                # Activate trailing stop if price moves in favorable direction by activation amount
                if not trailing_stop_active and row['close'] >= entry_price * (1 + trailing_activation):
                    trailing_stop_active = True
                    trailing_stop_level = max(stop_loss, row['close'] * (1 - trailing_step))
                
                # Update trailing stop if active
                if trailing_stop_active:
                    new_stop = row['close'] * (1 - trailing_step)
                    if new_stop > trailing_stop_level:
                        trailing_stop_level = new_stop
                    
                    # Check if trailing stop hit
                    if row['low'] <= trailing_stop_level:
                        result['result'] = 'win'  # Still a win if trailing stop hit
                        result['exit_price'] = trailing_stop_level
                        result['exit_date'] = i
                        result['return_pct'] = (trailing_stop_level - entry_price) / entry_price * 100
                        break
        
        else:  # bearish
            # Check if stop loss hit
            if row['high'] >= stop_loss:
                result['result'] = 'loss'
                result['exit_price'] = stop_loss
                result['exit_date'] = i
                result['return_pct'] = (entry_price - stop_loss) / entry_price * 100
                break
            
            # Check if take profit hit
            if row['low'] <= take_profit:
                result['result'] = 'win'
                result['exit_price'] = take_profit
                result['exit_date'] = i
                result['return_pct'] = (entry_price - take_profit) / entry_price * 100
                break
            
            # Check trailing stop if active
            if use_trailing_stop:
                # Activate trailing stop if price moves in favorable direction by activation amount
                if not trailing_stop_active and row['close'] <= entry_price * (1 - trailing_activation):
                    trailing_stop_active = True
                    trailing_stop_level = min(stop_loss, row['close'] * (1 + trailing_step))
                
                # Update trailing stop if active
                if trailing_stop_active:
                    new_stop = row['close'] * (1 + trailing_step)
                    if new_stop < trailing_stop_level:
                        trailing_stop_level = new_stop
                    
                    # Check if trailing stop hit
                    if row['high'] >= trailing_stop_level:
                        result['result'] = 'win'  # Still a win if trailing stop hit
                        result['exit_price'] = trailing_stop_level
                        result['exit_date'] = i
                        result['return_pct'] = (entry_price - trailing_stop_level) / entry_price * 100
                        break
    
    # If trade didn't hit stop loss or take profit, use last price
    if result['result'] is None:
        last_price = data['close'].iloc[-1]
        
        if direction == 'bullish':
            result['return_pct'] = (last_price - entry_price) / entry_price * 100
            if last_price > entry_price:
                result['result'] = 'win'
            else:
                result['result'] = 'loss'
        else:  # bearish
            result['return_pct'] = (entry_price - last_price) / entry_price * 100
            if last_price < entry_price:
                result['result'] = 'win'
            else:
                result['result'] = 'loss'
        
        result['exit_price'] = last_price
        result['exit_date'] = data.index[-1]
    
    return result 