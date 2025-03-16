import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('CandlestickPatternAnalyzer')

class CandlestickPatternAnalyzer:
    """
    Analyzes candlestick patterns in price data.
    """
    
    def __init__(self, data, body_threshold=0.6, doji_threshold=0.1):
        """
        Initialize the candlestick pattern analyzer.
        
        Args:
            data (pandas.DataFrame): DataFrame with OHLCV data
            body_threshold (float): Threshold for determining large bodies (0-1)
            doji_threshold (float): Threshold for determining doji (0-1)
        """
        self.data = data
        self.body_threshold = body_threshold
        self.doji_threshold = doji_threshold
        
        # Calculate candlestick properties
        self._calculate_properties()
        
        logger.info(f"Initialized Candlestick Pattern Analyzer with body threshold {body_threshold} and doji threshold {doji_threshold}")
    
    def _calculate_properties(self):
        """
        Calculate candlestick properties like body size, shadow size, etc.
        """
        # Make a copy to avoid modifying the original data
        self.candles = self.data.copy()
        
        # Calculate body size and direction
        self.candles['body_size'] = abs(self.candles['close'] - self.candles['open'])
        self.candles['body_pct'] = self.candles['body_size'] / (self.candles['high'] - self.candles['low'])
        self.candles['is_bullish'] = self.candles['close'] > self.candles['open']
        
        # Calculate upper and lower shadows
        self.candles['upper_shadow'] = self.candles.apply(
            lambda x: x['high'] - x['close'] if x['is_bullish'] else x['high'] - x['open'], 
            axis=1
        )
        self.candles['lower_shadow'] = self.candles.apply(
            lambda x: x['open'] - x['low'] if x['is_bullish'] else x['close'] - x['low'], 
            axis=1
        )
        
        # Calculate shadow percentages
        total_range = self.candles['high'] - self.candles['low']
        self.candles['upper_shadow_pct'] = self.candles['upper_shadow'] / total_range
        self.candles['lower_shadow_pct'] = self.candles['lower_shadow'] / total_range
        
        logger.info("Calculated candlestick properties")
    
    def detect_doji(self):
        """
        Detect doji candlestick patterns.
        
        Returns:
            pandas.Series: Boolean series indicating doji patterns
        """
        doji = self.candles['body_pct'] <= self.doji_threshold
        logger.info(f"Detected {doji.sum()} doji patterns")
        return doji
    
    def detect_hammer(self):
        """
        Detect hammer candlestick patterns.
        
        Returns:
            pandas.Series: Boolean series indicating hammer patterns
        """
        # Hammer has a small body at the top and a long lower shadow
        hammer = (
            (self.candles['body_pct'] <= 0.3) & 
            (self.candles['lower_shadow_pct'] >= 0.6) &
            (self.candles['upper_shadow_pct'] <= 0.1)
        )
        
        logger.info(f"Detected {hammer.sum()} hammer patterns")
        return hammer
    
    def detect_shooting_star(self):
        """
        Detect shooting star candlestick patterns.
        
        Returns:
            pandas.Series: Boolean series indicating shooting star patterns
        """
        # Shooting star has a small body at the bottom and a long upper shadow
        shooting_star = (
            (self.candles['body_pct'] <= 0.3) & 
            (self.candles['upper_shadow_pct'] >= 0.6) &
            (self.candles['lower_shadow_pct'] <= 0.1)
        )
        
        logger.info(f"Detected {shooting_star.sum()} shooting star patterns")
        return shooting_star
    
    def detect_engulfing(self):
        """
        Detect bullish and bearish engulfing patterns.
        
        Returns:
            tuple: (bullish_engulfing, bearish_engulfing) boolean series
        """
        bullish_engulfing = pd.Series(False, index=self.candles.index)
        bearish_engulfing = pd.Series(False, index=self.candles.index)
        
        # Need at least 2 candles
        if len(self.candles) < 2:
            logger.warning("Not enough data to detect engulfing patterns")
            return bullish_engulfing, bearish_engulfing
        
        # Detect bullish engulfing (current candle is bullish and engulfs previous bearish candle)
        for i in range(1, len(self.candles)):
            prev_candle = self.candles.iloc[i-1]
            curr_candle = self.candles.iloc[i]
            
            # Bullish engulfing
            if (not prev_candle['is_bullish'] and  # Previous candle is bearish
                curr_candle['is_bullish'] and      # Current candle is bullish
                curr_candle['open'] <= prev_candle['close'] and  # Current open below previous close
                curr_candle['close'] >= prev_candle['open']):    # Current close above previous open
                bullish_engulfing.iloc[i] = True
            
            # Bearish engulfing
            if (prev_candle['is_bullish'] and      # Previous candle is bullish
                not curr_candle['is_bullish'] and  # Current candle is bearish
                curr_candle['open'] >= prev_candle['close'] and  # Current open above previous close
                curr_candle['close'] <= prev_candle['open']):    # Current close below previous open
                bearish_engulfing.iloc[i] = True
        
        logger.info(f"Detected {bullish_engulfing.sum()} bullish engulfing and {bearish_engulfing.sum()} bearish engulfing patterns")
        return bullish_engulfing, bearish_engulfing
    
    def detect_morning_star(self):
        """
        Detect morning star patterns (bullish reversal).
        
        Returns:
            pandas.Series: Boolean series indicating morning star patterns
        """
        morning_star = pd.Series(False, index=self.candles.index)
        
        # Need at least 3 candles
        if len(self.candles) < 3:
            logger.warning("Not enough data to detect morning star patterns")
            return morning_star
        
        # Detect morning star (bearish candle, small body candle, bullish candle)
        for i in range(2, len(self.candles)):
            first_candle = self.candles.iloc[i-2]
            middle_candle = self.candles.iloc[i-1]
            last_candle = self.candles.iloc[i]
            
            if (not first_candle['is_bullish'] and  # First candle is bearish
                first_candle['body_pct'] >= self.body_threshold and  # First candle has large body
                middle_candle['body_pct'] <= 0.3 and  # Middle candle has small body
                last_candle['is_bullish'] and  # Last candle is bullish
                last_candle['body_pct'] >= self.body_threshold and  # Last candle has large body
                last_candle['close'] > (first_candle['open'] + first_candle['close']) / 2):  # Last close above first midpoint
                morning_star.iloc[i] = True
        
        logger.info(f"Detected {morning_star.sum()} morning star patterns")
        return morning_star
    
    def detect_evening_star(self):
        """
        Detect evening star patterns (bearish reversal).
        
        Returns:
            pandas.Series: Boolean series indicating evening star patterns
        """
        evening_star = pd.Series(False, index=self.candles.index)
        
        # Need at least 3 candles
        if len(self.candles) < 3:
            logger.warning("Not enough data to detect evening star patterns")
            return evening_star
        
        # Detect evening star (bullish candle, small body candle, bearish candle)
        for i in range(2, len(self.candles)):
            first_candle = self.candles.iloc[i-2]
            middle_candle = self.candles.iloc[i-1]
            last_candle = self.candles.iloc[i]
            
            if (first_candle['is_bullish'] and  # First candle is bullish
                first_candle['body_pct'] >= self.body_threshold and  # First candle has large body
                middle_candle['body_pct'] <= 0.3 and  # Middle candle has small body
                not last_candle['is_bullish'] and  # Last candle is bearish
                last_candle['body_pct'] >= self.body_threshold and  # Last candle has large body
                last_candle['close'] < (first_candle['open'] + first_candle['close']) / 2):  # Last close below first midpoint
                evening_star.iloc[i] = True
        
        logger.info(f"Detected {evening_star.sum()} evening star patterns")
        return evening_star
    
    def detect_pin_bar(self):
        """
        Detect pin bar patterns (long shadow in one direction).
        
        Returns:
            pandas.Series: Boolean series indicating pin bar patterns
        """
        # Pin bar has a small body and a long shadow in one direction
        pin_bar = (
            (self.candles['body_pct'] <= 0.25) & 
            ((self.candles['upper_shadow_pct'] >= 0.6) | (self.candles['lower_shadow_pct'] >= 0.6))
        )
        
        logger.info(f"Detected {pin_bar.sum()} pin bar patterns")
        return pin_bar
    
    def detect_all_patterns(self):
        """
        Detect all supported candlestick patterns.
        
        Returns:
            dict: Dictionary with pattern names as keys and boolean series as values
        """
        patterns = {
            'doji': self.detect_doji(),
            'hammer': self.detect_hammer(),
            'shooting_star': self.detect_shooting_star(),
            'pin_bar': self.detect_pin_bar()
        }
        
        # Patterns that require multiple candles
        if len(self.candles) >= 2:
            bullish_engulfing, bearish_engulfing = self.detect_engulfing()
            patterns['bullish_engulfing'] = bullish_engulfing
            patterns['bearish_engulfing'] = bearish_engulfing
        
        if len(self.candles) >= 3:
            patterns['morning_star'] = self.detect_morning_star()
            patterns['evening_star'] = self.detect_evening_star()
        
        return patterns
    
    def get_pattern_signals(self):
        """
        Get buy/sell signals based on candlestick patterns.
        
        Returns:
            pandas.DataFrame: DataFrame with buy and sell signals
        """
        patterns = self.detect_all_patterns()
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=self.candles.index)
        signals['buy'] = False
        signals['sell'] = False
        
        # Bullish patterns (buy signals)
        if 'bullish_engulfing' in patterns:
            signals['buy'] |= patterns['bullish_engulfing']
        
        if 'morning_star' in patterns:
            signals['buy'] |= patterns['morning_star']
        
        signals['buy'] |= patterns['hammer']
        
        # Bearish patterns (sell signals)
        if 'bearish_engulfing' in patterns:
            signals['sell'] |= patterns['bearish_engulfing']
        
        if 'evening_star' in patterns:
            signals['sell'] |= patterns['evening_star']
        
        signals['sell'] |= patterns['shooting_star']
        
        logger.info(f"Generated {signals['buy'].sum()} buy signals and {signals['sell'].sum()} sell signals")
        return signals 