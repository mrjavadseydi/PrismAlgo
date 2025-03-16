import pandas as pd
import numpy as np
import logging
from src.patterns.pattern_factory import PatternFactory

logger = logging.getLogger('HarmonicPatternAnalyzer')

class HarmonicPatternAnalyzer:
    """
    Analyzer for harmonic patterns in price data.
    """
    
    def __init__(self, data, config=None):
        """
        Initialize the harmonic pattern analyzer.
        
        Args:
            data (pandas.DataFrame): DataFrame with OHLCV data
            config (dict, optional): Configuration for the analyzer
        """
        self.data = data
        self.config = config or {}
        self.pattern_factory = PatternFactory()
        self.patterns = {}
        
        # Default configuration
        self.default_config = {
            'patterns': [
                'gartley',
                'butterfly',
                'bat',
                'crab',
                'shark',
                'cypher'
            ],
            'min_bars': 10,
            'max_bars': 100,
            'completion_threshold': 0.9
        }
        
        # Merge default config with provided config
        self._merge_config()
        
        logger.info("Initialized Harmonic Pattern Analyzer")
    
    def _merge_config(self):
        """
        Merge default configuration with provided configuration.
        """
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def analyze(self):
        """
        Analyze the data for harmonic patterns.
        
        Returns:
            dict: Dictionary of pattern name to detected patterns
        """
        self.patterns = {}
        
        # Get patterns to analyze
        patterns_to_analyze = self.config.get('patterns', self.default_config['patterns'])
        
        # Analyze each pattern
        for pattern_name in patterns_to_analyze:
            try:
                pattern = self.pattern_factory.get_pattern(pattern_name)
                if pattern:
                    self.patterns[pattern_name] = pattern.find_patterns(self.data)
                    logger.info(f"Found {len(self.patterns[pattern_name])} {pattern_name} patterns")
                else:
                    logger.warning(f"Pattern {pattern_name} not found")
            except Exception as e:
                logger.error(f"Error analyzing {pattern_name} pattern: {e}")
        
        return self.patterns
    
    def get_pattern_signals(self):
        """
        Get buy/sell signals from detected patterns.
        
        Returns:
            pandas.DataFrame: DataFrame with buy and sell signals
        """
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=self.data.index)
        signals['pattern'] = None
        signals['signal'] = None
        signals['strength'] = 0.0
        
        # If no patterns detected, return empty signals
        if not self.patterns:
            return signals
        
        # Process each pattern
        for pattern_name, patterns in self.patterns.items():
            if not patterns:
                continue
            
            # Check if patterns is a list or DataFrame
            if isinstance(patterns, pd.DataFrame):
                # Process DataFrame
                for idx, pattern in patterns.iterrows():
                    try:
                        self._process_pattern_signal(signals, pattern_name, pattern)
                    except Exception as e:
                        logger.error(f"Error generating signal for {pattern_name} pattern: {e}")
            else:
                # Process list
                for pattern in patterns:
                    try:
                        self._process_pattern_signal(signals, pattern_name, pattern)
                    except Exception as e:
                        logger.error(f"Error generating signal for {pattern_name} pattern: {e}")
        
        # Remove rows without signals
        signals = signals.dropna(subset=['signal'])
        
        logger.info(f"Generated {len(signals)} harmonic pattern signals")
        return signals

    def _process_pattern_signal(self, signals, pattern_name, pattern):
        """
        Process a single pattern and add its signal to the signals DataFrame.
        
        Args:
            signals (pandas.DataFrame): Signals DataFrame to update
            pattern_name (str): Name of the pattern
            pattern (dict): Pattern data
        """
        # Get pattern direction and completion
        direction = pattern.get('direction', 'bullish')
        completion = pattern.get('completion_percentage', 0) / 100.0
        
        # Only consider patterns that meet completion threshold
        if completion >= self.config.get('completion_threshold', 0.9):
            # Get pattern points
            points = pattern.get('points', {})
            
            # Get the last point (D) index
            if len(points) >= 4:
                d_point = points.get('D', None)
                if d_point and len(d_point) >= 2:
                    d_idx = d_point[0]
                    
                    # Check if index is valid
                    if isinstance(d_idx, (pd.Timestamp, str)):
                        if d_idx in signals.index:
                            # Set signal based on direction
                            if direction.lower() == 'bullish':
                                signals.loc[d_idx, 'signal'] = 'buy'
                            else:
                                signals.loc[d_idx, 'signal'] = 'sell'
                            
                            # Set pattern name and strength
                            signals.loc[d_idx, 'pattern'] = pattern_name
                            signals.loc[d_idx, 'strength'] = completion 