import numpy as np
import pandas as pd
import logging
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN
from scipy.stats import linregress

logger = logging.getLogger('PatternRecognition')

class PatternRecognition:
    """
    Advanced pattern recognition for price action analysis.
    This class implements algorithms to detect complex chart patterns
    that are not covered by basic candlestick analysis.
    """
    
    def __init__(self, data, config=None):
        """
        Initialize the pattern recognition module.
        
        Args:
            data (pandas.DataFrame): DataFrame with OHLCV data
            config (dict, optional): Configuration for pattern recognition
        """
        self.data = data
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'extrema_window': 10,
            'price_cluster_eps': 0.01,
            'price_cluster_min_samples': 2,
            'pattern_min_points': 4,
            'pattern_max_lookback': 100,
            'zz_deviation': 0.03,
            'channel_lookback': 20,
            'channel_deviation': 0.02
        }
        
        # Merge default config with provided config
        self._merge_config()
        
        # Check if data is valid
        if self.data is None or len(self.data) < 2 * self.config['extrema_window'] + 1:
            logger.warning(f"Not enough data points for pattern recognition. Need at least {2 * self.config['extrema_window'] + 1} data points.")
            self.high_points = pd.DataFrame(columns=['idx', 'price', 'date'])
            self.low_points = pd.DataFrame(columns=['idx', 'price', 'date'])
            self.highs = np.array([])
            self.lows = np.array([])
        else:
            # Precompute extrema points
            self._compute_extrema()
        
        logger.info("Initialized Pattern Recognition module")
    
    def _merge_config(self):
        """
        Merge default configuration with provided configuration.
        """
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def _compute_extrema(self):
        """
        Compute local extrema (peaks and troughs) in the price data.
        """
        try:
            window = self.config['extrema_window']
            
            # Find local maxima and minima
            self.highs = argrelextrema(self.data['high'].values, np.greater, order=window)[0]
            self.lows = argrelextrema(self.data['low'].values, np.less, order=window)[0]
            
            # Create DataFrames for extrema points
            if len(self.highs) > 0:
                self.high_points = pd.DataFrame({
                    'idx': self.highs,
                    'price': self.data['high'].iloc[self.highs].values,
                    'date': self.data.index[self.highs]
                })
            else:
                self.high_points = pd.DataFrame(columns=['idx', 'price', 'date'])
                
            if len(self.lows) > 0:
                self.low_points = pd.DataFrame({
                    'idx': self.lows,
                    'price': self.data['low'].iloc[self.lows].values,
                    'date': self.data.index[self.lows]
                })
            else:
                self.low_points = pd.DataFrame(columns=['idx', 'price', 'date'])
            
            logger.info(f"Computed {len(self.highs)} high points and {len(self.lows)} low points")
        except Exception as e:
            logger.error(f"Error computing extrema points: {e}")
            self.high_points = pd.DataFrame(columns=['idx', 'price', 'date'])
            self.low_points = pd.DataFrame(columns=['idx', 'price', 'date'])
            self.highs = np.array([])
            self.lows = np.array([])
    
    def detect_price_clusters(self):
        """
        Detect price clusters that may indicate support/resistance zones.
        
        Returns:
            dict: Dictionary with support and resistance clusters
        """
        if len(self.data) < 30:
            logger.warning("Not enough data to detect price clusters")
            return {'support': [], 'resistance': []}
        
        try:
            # Prepare data for clustering
            prices = self.data[['high', 'low']].values.flatten()
            prices = prices.reshape(-1, 1)
            
            # Normalize prices for clustering
            price_range = self.data['high'].max() - self.data['low'].min()
            eps = self.config['price_cluster_eps'] * price_range
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=self.config['price_cluster_min_samples']).fit(prices)
            
            # Get cluster centers
            clusters = {}
            for cluster_id in set(clustering.labels_):
                if cluster_id != -1:  # Skip noise points
                    cluster_prices = prices[clustering.labels_ == cluster_id]
                    clusters[cluster_id] = np.mean(cluster_prices)
            
            # Separate into support and resistance
            current_price = self.data['close'].iloc[-1]
            support = [float(price) for price in clusters.values() if float(price) < current_price]
            resistance = [float(price) for price in clusters.values() if float(price) > current_price]
            
            logger.info(f"Detected {len(support)} support clusters and {len(resistance)} resistance clusters")
            return {'support': support, 'resistance': resistance}
        except Exception as e:
            logger.error(f"Error detecting price clusters: {e}")
            return {'support': [], 'resistance': []}
    
    def detect_zigzag_patterns(self):
        """
        Detect zigzag patterns in the price data.
        
        Returns:
            list: List of zigzag patterns
        """
        if len(self.high_points) < 2 or len(self.low_points) < 2:
            logger.warning("Not enough extrema points to detect zigzag patterns")
            return []
        
        try:
            # Combine high and low points
            extrema = pd.concat([
                self.high_points.assign(type='high'),
                self.low_points.assign(type='low')
            ]).sort_values('idx')
            
            # Filter alternating high/low points
            zigzag_points = []
            current_type = None
            
            for _, point in extrema.iterrows():
                if current_type is None or point['type'] != current_type:
                    zigzag_points.append(point)
                    current_type = point['type']
            
            # Need at least 4 points for a zigzag pattern
            if len(zigzag_points) < 4:
                logger.warning("Not enough alternating points to detect zigzag patterns")
                return []
            
            # Find zigzag patterns
            patterns = []
            
            for i in range(len(zigzag_points) - 3):
                # Get 4 consecutive points
                points = zigzag_points[i:i+4]
                
                # Check if it's a valid zigzag (alternating high/low)
                if points[0]['type'] != points[1]['type'] and points[1]['type'] != points[2]['type'] and points[2]['type'] != points[3]['type']:
                    patterns.append({
                        'start_idx': int(points[0]['idx']),
                        'end_idx': int(points[3]['idx']),
                        'points': [(int(p['idx']), float(p['price']), p['type']) for p in points],
                        'start_date': points[0]['date'],
                        'end_date': points[3]['date']
                    })
            
            logger.info(f"Detected {len(patterns)} zigzag patterns")
            return patterns
        except Exception as e:
            logger.error(f"Error detecting zigzag patterns: {e}")
            return []
    
    def detect_channels(self):
        """
        Detect price channels (parallel support and resistance lines).
        
        Returns:
            list: List of detected channels
        """
        lookback = self.config['channel_lookback']
        deviation = self.config['channel_deviation']
        
        if len(self.data) < lookback:
            logger.warning(f"Not enough data to detect channels (need at least {lookback} points)")
            return []
        
        try:
            channels = []
            
            # Sliding window approach
            for start in range(0, len(self.data) - lookback, lookback // 2):
                end = start + lookback
                window_data = self.data.iloc[start:end]
                
                # Linear regression on highs
                x = np.arange(len(window_data))
                high_slope, high_intercept, _, _, _ = linregress(x, window_data['high'].values)
                
                # Linear regression on lows
                low_slope, low_intercept, _, _, _ = linregress(x, window_data['low'].values)
                
                # Check if slopes are similar (parallel lines)
                if abs(high_slope - low_slope) < deviation:
                    # Calculate channel boundaries
                    high_line = high_intercept + high_slope * x
                    low_line = low_intercept + low_slope * x
                    
                    # Calculate how well prices fit within the channel
                    highs_distance = np.abs(window_data['high'].values - high_line)
                    lows_distance = np.abs(window_data['low'].values - low_line)
                    
                    # Check if most prices are within the channel
                    if (np.mean(highs_distance) < deviation * window_data['high'].mean() and
                        np.mean(lows_distance) < deviation * window_data['low'].mean()):
                        channels.append({
                            'start_idx': int(start),
                            'end_idx': int(end - 1),
                            'high_slope': float(high_slope),
                            'high_intercept': float(high_intercept),
                            'low_slope': float(low_slope),
                            'low_intercept': float(low_intercept),
                            'start_date': self.data.index[start],
                            'end_date': self.data.index[end - 1]
                        })
            
            logger.info(f"Detected {len(channels)} price channels")
            return channels
        except Exception as e:
            logger.error(f"Error detecting channels: {e}")
            return []
    
    def detect_wedges(self):
        """
        Detect wedge patterns (converging support and resistance lines).
        
        Returns:
            list: List of detected wedges
        """
        lookback = self.config['channel_lookback']
        
        if len(self.data) < lookback:
            logger.warning(f"Not enough data to detect wedges (need at least {lookback} points)")
            return []
        
        try:
            wedges = []
            
            # Sliding window approach
            for start in range(0, len(self.data) - lookback, lookback // 2):
                end = start + lookback
                window_data = self.data.iloc[start:end]
                
                # Linear regression on highs
                x = np.arange(len(window_data))
                high_slope, high_intercept, _, _, _ = linregress(x, window_data['high'].values)
                
                # Linear regression on lows
                low_slope, low_intercept, _, _, _ = linregress(x, window_data['low'].values)
                
                # Check for converging lines (wedge)
                if (high_slope < 0 and low_slope > 0) or (high_slope > 0 and low_slope < 0):
                    # Calculate wedge boundaries
                    high_line = high_intercept + high_slope * x
                    low_line = low_intercept + low_slope * x
                    
                    # Calculate how well prices fit within the wedge
                    highs_distance = np.abs(window_data['high'].values - high_line)
                    lows_distance = np.abs(window_data['low'].values - low_line)
                    
                    # Check if most prices are within the wedge
                    if (np.mean(highs_distance) < 0.03 * window_data['high'].mean() and
                        np.mean(lows_distance) < 0.03 * window_data['low'].mean()):
                        
                        # Determine wedge type
                        if high_slope < 0 and low_slope > 0:
                            wedge_type = 'converging'
                        else:
                            wedge_type = 'diverging'
                        
                        wedges.append({
                            'start_idx': int(start),
                            'end_idx': int(end - 1),
                            'high_slope': float(high_slope),
                            'high_intercept': float(high_intercept),
                            'low_slope': float(low_slope),
                            'low_intercept': float(low_intercept),
                            'type': wedge_type,
                            'start_date': self.data.index[start],
                            'end_date': self.data.index[end - 1]
                        })
            
            logger.info(f"Detected {len(wedges)} wedge patterns")
            return wedges
        except Exception as e:
            logger.error(f"Error detecting wedges: {e}")
            return []
    
    def detect_double_top_bottom(self):
        """
        Detect double top and double bottom patterns.
        
        Returns:
            dict: Dictionary with double top and double bottom patterns
        """
        if len(self.high_points) < 2 or len(self.low_points) < 2:
            logger.warning("Not enough extrema points to detect double top/bottom patterns")
            return {'double_top': [], 'double_bottom': []}
        
        try:
            double_tops = []
            double_bottoms = []
            
            # Detect double tops
            for i in range(len(self.high_points) - 1):
                for j in range(i + 1, len(self.high_points)):
                    # Check if the two highs are similar in price
                    price_diff = abs(self.high_points.iloc[i]['price'] - self.high_points.iloc[j]['price'])
                    avg_price = (self.high_points.iloc[i]['price'] + self.high_points.iloc[j]['price']) / 2
                    
                    if price_diff / avg_price < 0.02:  # 2% tolerance
                        # Check if there's a significant drop between the two tops
                        idx_between = self.data.index[(self.data.index >= self.high_points.iloc[i]['date']) & 
                                                     (self.data.index <= self.high_points.iloc[j]['date'])]
                        
                        if len(idx_between) > 0:
                            min_between = self.data.loc[idx_between, 'low'].min()
                            drop = (avg_price - min_between) / avg_price
                            
                            if drop > 0.03:  # At least 3% drop between tops
                                double_tops.append({
                                    'first_top_idx': int(self.high_points.iloc[i]['idx']),
                                    'second_top_idx': int(self.high_points.iloc[j]['idx']),
                                    'first_top_price': float(self.high_points.iloc[i]['price']),
                                    'second_top_price': float(self.high_points.iloc[j]['price']),
                                    'first_date': self.high_points.iloc[i]['date'],
                                    'second_date': self.high_points.iloc[j]['date']
                                })
            
            # Detect double bottoms
            for i in range(len(self.low_points) - 1):
                for j in range(i + 1, len(self.low_points)):
                    # Check if the two lows are similar in price
                    price_diff = abs(self.low_points.iloc[i]['price'] - self.low_points.iloc[j]['price'])
                    avg_price = (self.low_points.iloc[i]['price'] + self.low_points.iloc[j]['price']) / 2
                    
                    if price_diff / avg_price < 0.02:  # 2% tolerance
                        # Check if there's a significant rise between the two bottoms
                        idx_between = self.data.index[(self.data.index >= self.low_points.iloc[i]['date']) & 
                                                     (self.data.index <= self.low_points.iloc[j]['date'])]
                        
                        if len(idx_between) > 0:
                            max_between = self.data.loc[idx_between, 'high'].max()
                            rise = (max_between - avg_price) / avg_price
                            
                            if rise > 0.03:  # At least 3% rise between bottoms
                                double_bottoms.append({
                                    'first_bottom_idx': int(self.low_points.iloc[i]['idx']),
                                    'second_bottom_idx': int(self.low_points.iloc[j]['idx']),
                                    'first_bottom_price': float(self.low_points.iloc[i]['price']),
                                    'second_bottom_price': float(self.low_points.iloc[j]['price']),
                                    'first_date': self.low_points.iloc[i]['date'],
                                    'second_date': self.low_points.iloc[j]['date']
                                })
            
            logger.info(f"Detected {len(double_tops)} double tops and {len(double_bottoms)} double bottoms")
            return {'double_top': double_tops, 'double_bottom': double_bottoms}
        except Exception as e:
            logger.error(f"Error detecting double top/bottom patterns: {e}")
            return {'double_top': [], 'double_bottom': []}
    
    def get_pattern_signals(self):
        """
        Generate trading signals based on detected patterns.
        
        Returns:
            pandas.DataFrame: DataFrame with buy and sell signals
        """
        try:
            # Initialize signals DataFrame
            signals = pd.DataFrame(index=self.data.index)
            signals['buy'] = False
            signals['sell'] = False
            
            # Get double tops and bottoms
            double_patterns = self.detect_double_top_bottom()
            
            # Generate signals for double tops (sell)
            for pattern in double_patterns['double_top']:
                # Sell signal after the second top
                idx = pattern['second_top_idx']
                if idx < len(self.data):
                    signals.loc[self.data.index[idx], 'sell'] = True
            
            # Generate signals for double bottoms (buy)
            for pattern in double_patterns['double_bottom']:
                # Buy signal after the second bottom
                idx = pattern['second_bottom_idx']
                if idx < len(self.data):
                    signals.loc[self.data.index[idx], 'buy'] = True
            
            # Get wedges
            wedges = self.detect_wedges()
            
            # Generate signals for wedges
            for wedge in wedges:
                # Signal at the end of the wedge
                idx = wedge['end_idx']
                if idx < len(self.data):
                    # Converging wedges can break either way, but often break opposite to the direction
                    if wedge['type'] == 'converging':
                        if wedge['high_slope'] < wedge['low_slope']:  # Bearish wedge
                            signals.loc[self.data.index[idx], 'sell'] = True
                        else:  # Bullish wedge
                            signals.loc[self.data.index[idx], 'buy'] = True
            
            # Get channels
            channels = self.detect_channels()
            
            # Generate signals for channels
            for channel in channels:
                idx = channel['end_idx']
                if idx < len(self.data):
                    # Buy at lower channel boundary, sell at upper boundary
                    current_price = self.data['close'].iloc[idx]
                    high_boundary = channel['high_intercept'] + channel['high_slope'] * (idx - channel['start_idx'])
                    low_boundary = channel['low_intercept'] + channel['low_slope'] * (idx - channel['start_idx'])
                    
                    # If price is near the lower boundary
                    if abs(current_price - low_boundary) / low_boundary < 0.01:
                        signals.loc[self.data.index[idx], 'buy'] = True
                    
                    # If price is near the upper boundary
                    if abs(current_price - high_boundary) / high_boundary < 0.01:
                        signals.loc[self.data.index[idx], 'sell'] = True
            
            logger.info(f"Generated {signals['buy'].sum()} buy signals and {signals['sell'].sum()} sell signals from pattern recognition")
            return signals
        except Exception as e:
            logger.error(f"Error generating pattern signals: {e}")
            # Return empty signals DataFrame
            signals = pd.DataFrame(index=self.data.index)
            signals['buy'] = False
            signals['sell'] = False
            return signals 