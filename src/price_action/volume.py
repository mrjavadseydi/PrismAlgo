import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('VolumeAnalyzer')

class VolumeAnalyzer:
    """
    Analyzes volume patterns and their relationship with price movements.
    """
    
    def __init__(self, data, window_size=20):
        """
        Initialize the volume analyzer.
        
        Args:
            data (pandas.DataFrame): DataFrame with OHLCV data
            window_size (int): Window size for moving averages and other calculations
        """
        self.data = data
        self.window_size = window_size
        
        # Calculate volume indicators
        self._calculate_indicators()
        
        logger.info(f"Initialized Volume Analyzer with window size {window_size}")
    
    def _calculate_indicators(self):
        """
        Calculate volume-based indicators.
        """
        # Make a copy to avoid modifying the original data
        self.volume_data = self.data.copy()
        
        # Calculate price change
        self.volume_data['price_change'] = self.volume_data['close'].pct_change()
        
        # Calculate volume change
        self.volume_data['volume_change'] = self.volume_data['volume'].pct_change()
        
        # Calculate volume moving average
        self.volume_data['volume_ma'] = self.volume_data['volume'].rolling(window=self.window_size).mean()
        
        # Calculate relative volume (current volume / average volume)
        self.volume_data['relative_volume'] = self.volume_data['volume'] / self.volume_data['volume_ma']
        
        # Calculate on-balance volume (OBV)
        self.volume_data['obv'] = 0
        for i in range(1, len(self.volume_data)):
            prev_obv = self.volume_data['obv'].iloc[i-1]
            if self.volume_data['close'].iloc[i] > self.volume_data['close'].iloc[i-1]:
                self.volume_data.loc[self.volume_data.index[i], 'obv'] = prev_obv + self.volume_data['volume'].iloc[i]
            elif self.volume_data['close'].iloc[i] < self.volume_data['close'].iloc[i-1]:
                self.volume_data.loc[self.volume_data.index[i], 'obv'] = prev_obv - self.volume_data['volume'].iloc[i]
            else:
                self.volume_data.loc[self.volume_data.index[i], 'obv'] = prev_obv
        
        logger.info("Calculated volume indicators")
    
    def detect_volume_spikes(self, threshold=2.0):
        """
        Detect significant volume spikes.
        
        Args:
            threshold (float): Threshold for volume spike detection (relative to moving average)
            
        Returns:
            pandas.Series: Boolean series indicating volume spikes
        """
        volume_spikes = self.volume_data['relative_volume'] >= threshold
        logger.info(f"Detected {volume_spikes.sum()} volume spikes with threshold {threshold}")
        return volume_spikes
    
    def detect_volume_climax(self, lookback=5, threshold=3.0):
        """
        Detect volume climax (extremely high volume that may signal exhaustion).
        
        Args:
            lookback (int): Number of periods to look back
            threshold (float): Threshold for volume climax detection
            
        Returns:
            pandas.Series: Boolean series indicating volume climax
        """
        # Volume climax is an extreme volume spike that's significantly higher than recent volumes
        volume_climax = pd.Series(False, index=self.volume_data.index)
        
        for i in range(lookback, len(self.volume_data)):
            recent_volumes = self.volume_data['volume'].iloc[i-lookback:i]
            current_volume = self.volume_data['volume'].iloc[i]
            
            if current_volume > threshold * recent_volumes.mean():
                volume_climax.iloc[i] = True
        
        logger.info(f"Detected {volume_climax.sum()} volume climax events")
        return volume_climax
    
    def detect_volume_divergence(self, window=10):
        """
        Detect divergence between price and volume.
        
        Args:
            window (int): Window size for divergence detection
            
        Returns:
            tuple: (bullish_divergence, bearish_divergence) boolean series
        """
        bullish_divergence = pd.Series(False, index=self.volume_data.index)
        bearish_divergence = pd.Series(False, index=self.volume_data.index)
        
        # Need at least window+1 data points
        if len(self.volume_data) <= window:
            logger.warning(f"Not enough data to detect volume divergence. Need more than {window} data points.")
            return bullish_divergence, bearish_divergence
        
        # Calculate price and volume trends
        for i in range(window, len(self.volume_data)):
            price_trend = self.volume_data['close'].iloc[i] - self.volume_data['close'].iloc[i-window]
            volume_trend = self.volume_data['volume'].iloc[i] - self.volume_data['volume'].iloc[i-window]
            
            # Bullish divergence: price making lower lows but volume making higher lows
            if price_trend < 0 and volume_trend > 0:
                bullish_divergence.iloc[i] = True
            
            # Bearish divergence: price making higher highs but volume making lower highs
            if price_trend > 0 and volume_trend < 0:
                bearish_divergence.iloc[i] = True
        
        logger.info(f"Detected {bullish_divergence.sum()} bullish divergences and {bearish_divergence.sum()} bearish divergences")
        return bullish_divergence, bearish_divergence
    
    def detect_volume_confirmation(self):
        """
        Detect when volume confirms price movement.
        
        Returns:
            pandas.Series: Boolean series indicating volume confirmation
        """
        # Volume confirms price when they move in the same direction
        confirmation = (
            (self.volume_data['price_change'] > 0) & (self.volume_data['volume_change'] > 0) |
            (self.volume_data['price_change'] < 0) & (self.volume_data['volume_change'] > 0)
        )
        
        logger.info(f"Detected {confirmation.sum()} volume confirmations")
        return confirmation
    
    def detect_churn(self, threshold=0.5):
        """
        Detect churning (high volume with little price movement).
        
        Args:
            threshold (float): Threshold for price movement significance
            
        Returns:
            pandas.Series: Boolean series indicating churning
        """
        # Churning is high volume with little price movement
        churn = (
            (self.volume_data['relative_volume'] > 1.5) & 
            (abs(self.volume_data['price_change']) < threshold * self.volume_data['price_change'].rolling(window=self.window_size).std())
        )
        
        logger.info(f"Detected {churn.sum()} churning events")
        return churn
    
    def get_volume_signals(self):
        """
        Get buy/sell signals based on volume analysis.
        
        Returns:
            pandas.DataFrame: DataFrame with buy and sell signals
        """
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=self.volume_data.index)
        signals['buy'] = False
        signals['sell'] = False
        
        # Detect volume patterns
        volume_spikes = self.detect_volume_spikes()
        bullish_divergence, bearish_divergence = self.detect_volume_divergence()
        volume_confirmation = self.detect_volume_confirmation()
        
        # Generate buy signals
        signals['buy'] |= (
            bullish_divergence & 
            volume_spikes & 
            (self.volume_data['price_change'] > 0)
        )
        
        # Generate sell signals
        signals['sell'] |= (
            bearish_divergence & 
            volume_spikes & 
            (self.volume_data['price_change'] < 0)
        )
        
        logger.info(f"Generated {signals['buy'].sum()} buy signals and {signals['sell'].sum()} sell signals based on volume analysis")
        return signals
    
    def get_obv_trend(self, window=20):
        """
        Get the trend of On-Balance Volume (OBV).
        
        Args:
            window (int): Window size for trend calculation
            
        Returns:
            pandas.Series: Series with trend values (1 for uptrend, -1 for downtrend, 0 for no trend)
        """
        obv_ma = self.volume_data['obv'].rolling(window=window).mean()
        
        trend = pd.Series(0, index=self.volume_data.index)
        trend[self.volume_data['obv'] > obv_ma] = 1  # Uptrend
        trend[self.volume_data['obv'] < obv_ma] = -1  # Downtrend
        
        return trend 