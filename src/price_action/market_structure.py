import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('MarketStructureAnalyzer')

class MarketStructureAnalyzer:
    """
    Analyzes market structure including support/resistance levels, trends, and swing points.
    """
    
    def __init__(self, data, window_size=20, threshold=0.02):
        """
        Initialize the market structure analyzer.
        
        Args:
            data (pandas.DataFrame): DataFrame with OHLCV data
            window_size (int): Window size for detecting swing points
            threshold (float): Threshold for support/resistance significance
        """
        self.data = data
        self.window_size = window_size
        self.threshold = threshold
        self.support_levels = []
        self.resistance_levels = []
        self.swing_highs = []
        self.swing_lows = []
        self.trend = None
        
        logger.info(f"Initialized Market Structure Analyzer with window size {window_size}")
    
    def detect_swing_points(self):
        """
        Detect swing highs and swing lows in the price data.
        
        Returns:
            tuple: Lists of swing highs and swing lows as (index, price) tuples
        """
        highs = []
        lows = []
        
        # Need at least 2*window_size+1 data points
        if len(self.data) < 2 * self.window_size + 1:
            logger.warning(f"Not enough data points to detect swing points. Need at least {2 * self.window_size + 1}, got {len(self.data)}")
            return highs, lows
        
        # Detect swing highs
        for i in range(self.window_size, len(self.data) - self.window_size):
            if self.is_swing_high(i):
                highs.append((i, self.data['high'].iloc[i]))
        
        # Detect swing lows
        for i in range(self.window_size, len(self.data) - self.window_size):
            if self.is_swing_low(i):
                lows.append((i, self.data['low'].iloc[i]))
        
        self.swing_highs = highs
        self.swing_lows = lows
        
        logger.info(f"Detected {len(highs)} swing highs and {len(lows)} swing lows")
        return highs, lows
    
    def is_swing_high(self, index):
        """
        Check if the given index is a swing high.
        
        Args:
            index (int): Index to check
            
        Returns:
            bool: True if the index is a swing high, False otherwise
        """
        window_left = self.data['high'].iloc[index - self.window_size:index]
        window_right = self.data['high'].iloc[index + 1:index + self.window_size + 1]
        current_high = self.data['high'].iloc[index]
        
        return current_high > window_left.max() and current_high > window_right.max()
    
    def is_swing_low(self, index):
        """
        Check if the given index is a swing low.
        
        Args:
            index (int): Index to check
            
        Returns:
            bool: True if the index is a swing low, False otherwise
        """
        window_left = self.data['low'].iloc[index - self.window_size:index]
        window_right = self.data['low'].iloc[index + 1:index + self.window_size + 1]
        current_low = self.data['low'].iloc[index]
        
        return current_low < window_left.min() and current_low < window_right.min()
    
    def detect_support_resistance(self):
        """
        Detect support and resistance levels based on swing points.
        
        Returns:
            tuple: Lists of support and resistance levels
        """
        if not self.swing_highs or not self.swing_lows:
            self.detect_swing_points()
        
        # Group nearby swing highs to identify resistance levels
        resistance_levels = self._cluster_price_levels([price for _, price in self.swing_highs])
        
        # Group nearby swing lows to identify support levels
        support_levels = self._cluster_price_levels([price for _, price in self.swing_lows])
        
        self.support_levels = support_levels
        self.resistance_levels = resistance_levels
        
        logger.info(f"Detected {len(support_levels)} support levels and {len(resistance_levels)} resistance levels")
        return support_levels, resistance_levels
    
    def _cluster_price_levels(self, prices, cluster_threshold=0.01):
        """
        Cluster nearby price levels.
        
        Args:
            prices (list): List of price levels
            cluster_threshold (float): Threshold for clustering
            
        Returns:
            list: Clustered price levels
        """
        if not prices:
            return []
        
        # Sort prices
        sorted_prices = sorted(prices)
        
        # Initialize clusters
        clusters = [[sorted_prices[0]]]
        
        # Cluster prices
        for price in sorted_prices[1:]:
            last_cluster = clusters[-1]
            last_price = last_cluster[-1]
            
            # If price is close to the last price, add to the same cluster
            if abs(price - last_price) / last_price < cluster_threshold:
                last_cluster.append(price)
            else:
                # Otherwise, create a new cluster
                clusters.append([price])
        
        # Calculate average price for each cluster
        return [sum(cluster) / len(cluster) for cluster in clusters]
    
    def analyze_trend(self, lookback=10):
        """
        Analyze the current market trend.
        
        Args:
            lookback (int): Number of swing points to consider
            
        Returns:
            str: 'uptrend', 'downtrend', or 'sideways'
        """
        if not self.swing_highs or not self.swing_lows:
            self.detect_swing_points()
        
        # Get recent swing highs and lows
        recent_highs = self.swing_highs[-min(lookback, len(self.swing_highs)):]
        recent_lows = self.swing_lows[-min(lookback, len(self.swing_lows)):]
        
        if not recent_highs or not recent_lows:
            logger.warning("Not enough swing points to analyze trend")
            self.trend = 'sideways'
            return self.trend
        
        # Check for higher highs and higher lows (uptrend)
        higher_highs = all(recent_highs[i][1] > recent_highs[i-1][1] for i in range(1, len(recent_highs)))
        higher_lows = all(recent_lows[i][1] > recent_lows[i-1][1] for i in range(1, len(recent_lows)))
        
        # Check for lower highs and lower lows (downtrend)
        lower_highs = all(recent_highs[i][1] < recent_highs[i-1][1] for i in range(1, len(recent_highs)))
        lower_lows = all(recent_lows[i][1] < recent_lows[i-1][1] for i in range(1, len(recent_lows)))
        
        if higher_highs and higher_lows:
            self.trend = 'uptrend'
        elif lower_highs and lower_lows:
            self.trend = 'downtrend'
        else:
            self.trend = 'sideways'
        
        logger.info(f"Current market trend: {self.trend}")
        return self.trend
    
    def get_key_levels(self):
        """
        Get key price levels (support and resistance).
        
        Returns:
            dict: Dictionary with support and resistance levels
        """
        if not self.support_levels or not self.resistance_levels:
            self.detect_support_resistance()
        
        return {
            'support': self.support_levels,
            'resistance': self.resistance_levels
        }
    
    def is_near_key_level(self, price, threshold=0.01):
        """
        Check if the given price is near a key level.
        
        Args:
            price (float): Price to check
            threshold (float): Threshold for proximity
            
        Returns:
            tuple: (bool, str) indicating if near a key level and the type of level
        """
        if not self.support_levels or not self.resistance_levels:
            self.detect_support_resistance()
        
        # Check support levels
        for level in self.support_levels:
            if abs(price - level) / level < threshold:
                return True, 'support'
        
        # Check resistance levels
        for level in self.resistance_levels:
            if abs(price - level) / level < threshold:
                return True, 'resistance'
        
        return False, None 