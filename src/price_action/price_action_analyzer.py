import pandas as pd
import logging
from src.price_action.market_structure import MarketStructureAnalyzer
from src.price_action.candlestick import CandlestickPatternAnalyzer
from src.price_action.volume import VolumeAnalyzer
from src.price_action.pattern_recognition import PatternRecognition

logger = logging.getLogger('PriceActionAnalyzer')

class PriceActionAnalyzer:
    """
    Main class for price action analysis that combines market structure,
    candlestick patterns, volume analysis, and advanced pattern recognition.
    """
    
    def __init__(self, data, config=None):
        """
        Initialize the price action analyzer.
        
        Args:
            data (pandas.DataFrame): DataFrame with OHLCV data
            config (dict, optional): Configuration for the analyzers
        """
        self.data = data
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'market_structure': {
                'enabled': True,
                'window_size': 20,
                'threshold': 0.02
            },
            'candlestick': {
                'enabled': True,
                'body_threshold': 0.6,
                'doji_threshold': 0.1
            },
            'volume': {
                'enabled': True,
                'window_size': 20
            },
            'pattern_recognition': {
                'enabled': True,
                'extrema_window': 10,
                'price_cluster_eps': 0.01,
                'price_cluster_min_samples': 2,
                'channel_lookback': 20,
                'channel_deviation': 0.02
            }
        }
        
        # Merge default config with provided config
        self._merge_config()
        
        # Initialize analyzers
        self._initialize_analyzers()
        
        logger.info("Initialized Price Action Analyzer")
    
    def _merge_config(self):
        """
        Merge default configuration with provided configuration.
        """
        for category in self.default_config:
            if category not in self.config:
                self.config[category] = self.default_config[category]
            else:
                for key, value in self.default_config[category].items():
                    if key not in self.config[category]:
                        self.config[category][key] = value
    
    def _initialize_analyzers(self):
        """
        Initialize the individual analyzers based on configuration.
        """
        self.analyzers = {}
        
        # Initialize market structure analyzer
        if self.config['market_structure']['enabled']:
            self.analyzers['market_structure'] = MarketStructureAnalyzer(
                self.data,
                window_size=self.config['market_structure']['window_size'],
                threshold=self.config['market_structure']['threshold']
            )
        
        # Initialize candlestick pattern analyzer
        if self.config['candlestick']['enabled']:
            self.analyzers['candlestick'] = CandlestickPatternAnalyzer(
                self.data,
                body_threshold=self.config['candlestick']['body_threshold'],
                doji_threshold=self.config['candlestick']['doji_threshold']
            )
        
        # Initialize volume analyzer
        if self.config['volume']['enabled']:
            self.analyzers['volume'] = VolumeAnalyzer(
                self.data,
                window_size=self.config['volume']['window_size']
            )
            
        # Initialize pattern recognition analyzer
        if self.config['pattern_recognition']['enabled']:
            self.analyzers['pattern_recognition'] = PatternRecognition(
                self.data,
                config=self.config['pattern_recognition']
            )
    
    def analyze(self):
        """
        Perform comprehensive price action analysis.
        
        Returns:
            dict: Analysis results
        """
        results = {}
        
        # Market structure analysis
        if 'market_structure' in self.analyzers:
            ms_analyzer = self.analyzers['market_structure']
            ms_analyzer.detect_swing_points()
            ms_analyzer.detect_support_resistance()
            ms_analyzer.analyze_trend()
            
            results['market_structure'] = {
                'support_levels': ms_analyzer.support_levels,
                'resistance_levels': ms_analyzer.resistance_levels,
                'trend': ms_analyzer.trend,
                'swing_highs': ms_analyzer.swing_highs,
                'swing_lows': ms_analyzer.swing_lows
            }
        
        # Candlestick pattern analysis
        if 'candlestick' in self.analyzers:
            cs_analyzer = self.analyzers['candlestick']
            patterns = cs_analyzer.detect_all_patterns()
            pattern_signals = cs_analyzer.get_pattern_signals()
            
            results['candlestick'] = {
                'patterns': patterns,
                'signals': pattern_signals
            }
        
        # Volume analysis
        if 'volume' in self.analyzers:
            vol_analyzer = self.analyzers['volume']
            volume_spikes = vol_analyzer.detect_volume_spikes()
            bullish_div, bearish_div = vol_analyzer.detect_volume_divergence()
            volume_confirmation = vol_analyzer.detect_volume_confirmation()
            volume_signals = vol_analyzer.get_volume_signals()
            
            results['volume'] = {
                'volume_spikes': volume_spikes,
                'bullish_divergence': bullish_div,
                'bearish_divergence': bearish_div,
                'volume_confirmation': volume_confirmation,
                'signals': volume_signals
            }
            
        # Pattern recognition analysis
        if 'pattern_recognition' in self.analyzers:
            pr_analyzer = self.analyzers['pattern_recognition']
            
            # Detect price clusters (support/resistance zones)
            price_clusters = pr_analyzer.detect_price_clusters()
            
            # Detect chart patterns
            zigzag_patterns = pr_analyzer.detect_zigzag_patterns()
            channels = pr_analyzer.detect_channels()
            wedges = pr_analyzer.detect_wedges()
            double_patterns = pr_analyzer.detect_double_top_bottom()
            
            # Get pattern signals
            pattern_signals = pr_analyzer.get_pattern_signals()
            
            results['pattern_recognition'] = {
                'price_clusters': price_clusters,
                'zigzag_patterns': zigzag_patterns,
                'channels': channels,
                'wedges': wedges,
                'double_patterns': double_patterns,
                'signals': pattern_signals
            }
        
        logger.info("Completed price action analysis")
        return results
    
    def get_combined_signals(self):
        """
        Get combined buy/sell signals from all analyzers.
        
        Returns:
            pandas.DataFrame: DataFrame with combined buy and sell signals
        """
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=self.data.index)
        signals['buy'] = False
        signals['sell'] = False
        
        # Collect signals from each analyzer
        if 'candlestick' in self.analyzers:
            cs_signals = self.analyzers['candlestick'].get_pattern_signals()
            signals['buy'] |= cs_signals['buy']
            signals['sell'] |= cs_signals['sell']
        
        if 'volume' in self.analyzers:
            vol_signals = self.analyzers['volume'].get_volume_signals()
            signals['buy'] |= vol_signals['buy']
            signals['sell'] |= vol_signals['sell']
            
        if 'pattern_recognition' in self.analyzers:
            pr_signals = self.analyzers['pattern_recognition'].get_pattern_signals()
            signals['buy'] |= pr_signals['buy']
            signals['sell'] |= pr_signals['sell']
        
        # Add market structure context
        if 'market_structure' in self.analyzers:
            ms_analyzer = self.analyzers['market_structure']
            
            # Filter signals based on trend
            trend = ms_analyzer.analyze_trend()
            
            # In uptrend, prioritize buy signals and filter sell signals
            if trend == 'uptrend':
                signals['sell'] = signals['sell'] & (signals['sell'].rolling(window=3).sum() >= 2)
            
            # In downtrend, prioritize sell signals and filter buy signals
            elif trend == 'downtrend':
                signals['buy'] = signals['buy'] & (signals['buy'].rolling(window=3).sum() >= 2)
            
            # Check if price is near key levels
            for i in range(len(signals)):
                if i < len(self.data):
                    current_price = self.data['close'].iloc[i]
                    is_near, level_type = ms_analyzer.is_near_key_level(current_price)
                    
                    # Strengthen signals near key levels
                    if is_near:
                        if level_type == 'support' and signals['buy'].iloc[i]:
                            # Stronger buy signal at support
                            pass  # Already True, just keeping it
                        elif level_type == 'resistance' and signals['sell'].iloc[i]:
                            # Stronger sell signal at resistance
                            pass  # Already True, just keeping it
        
        # Add confluence scoring - signals with multiple confirmations are stronger
        signal_strength = pd.DataFrame(index=self.data.index)
        signal_strength['buy_strength'] = 0
        signal_strength['sell_strength'] = 0
        
        # Count how many analyzers confirm each signal
        for analyzer_name, analyzer in self.analyzers.items():
            if hasattr(analyzer, 'get_pattern_signals'):
                analyzer_signals = analyzer.get_pattern_signals()
                signal_strength['buy_strength'] += analyzer_signals['buy'].astype(int)
                signal_strength['sell_strength'] += analyzer_signals['sell'].astype(int)
        
        # Add strength information to the signals
        signals['buy_strength'] = signal_strength['buy_strength']
        signals['sell_strength'] = signal_strength['sell_strength']
        
        # Filter weak signals (optional)
        # signals['buy'] = signals['buy'] & (signals['buy_strength'] >= 2)
        # signals['sell'] = signals['sell'] & (signals['sell_strength'] >= 2)
        
        logger.info(f"Generated {signals['buy'].sum()} combined buy signals and {signals['sell'].sum()} combined sell signals")
        return signals
    
    def _calculate_performance_metrics(self, signals):
        """
        Calculate performance metrics for price action signals.
        
        Args:
            signals (pandas.DataFrame): DataFrame with buy and sell signals
            
        Returns:
            dict: Dictionary with performance metrics
        """
        try:
            # Initialize metrics
            metrics = {
                'win_rate': 0.0,
                'profit_factor': 1.0,
                'total_return': 0.0,
                'max_drawdown': 0.0
            }
            
            # If no signals, return default metrics
            if not signals['buy'].any() and not signals['sell'].any():
                return metrics
            
            # Create a copy of the data to avoid modifying the original
            data = self.data.copy()
            
            # Add signals to the data
            data['buy_signal'] = signals['buy']
            data['sell_signal'] = signals['sell']
            
            # Initialize position and equity columns
            data['position'] = 0
            data['equity'] = 0.0
            
            # Set initial position based on first signal
            in_position = False
            entry_price = 0.0
            position_type = None  # 'long' or 'short'
            trades = []
            
            # Simulate trades
            for i in range(1, len(data)):
                # Check for buy signal
                if data['buy_signal'].iloc[i] and not in_position:
                    in_position = True
                    position_type = 'long'
                    entry_price = data['close'].iloc[i]
                    data.loc[data.index[i], 'position'] = 1
                
                # Check for sell signal
                elif data['sell_signal'].iloc[i] and not in_position:
                    in_position = True
                    position_type = 'short'
                    entry_price = data['close'].iloc[i]
                    data.loc[data.index[i], 'position'] = -1
                
                # Check for exit (opposite signal)
                elif (data['sell_signal'].iloc[i] and in_position and position_type == 'long') or \
                     (data['buy_signal'].iloc[i] and in_position and position_type == 'short'):
                    exit_price = data['close'].iloc[i]
                    
                    # Calculate profit/loss
                    if position_type == 'long':
                        profit_pct = (exit_price - entry_price) / entry_price
                    else:  # short
                        profit_pct = (entry_price - exit_price) / entry_price
                    
                    # Record trade
                    trades.append({
                        'entry_date': data.index[i-1],
                        'exit_date': data.index[i],
                        'position_type': position_type,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit_pct': profit_pct
                    })
                    
                    # Reset position
                    in_position = False
                    entry_price = 0.0
                    position_type = None
                    data.loc[data.index[i], 'position'] = 0
            
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
                # Win rate
                winning_trades = sum(1 for trade in trades if trade['profit_pct'] > 0)
                metrics['win_rate'] = winning_trades / len(trades)
                
                # Profit factor
                gross_profit = sum(trade['profit_pct'] for trade in trades if trade['profit_pct'] > 0)
                gross_loss = abs(sum(trade['profit_pct'] for trade in trades if trade['profit_pct'] < 0))
                metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                # Total return
                metrics['total_return'] = (current_equity - initial_equity) / initial_equity
                
                # Max drawdown
                peak = initial_equity
                drawdown = 0
                max_drawdown = 0
                
                for equity in equity_curve:
                    if equity > peak:
                        peak = equity
                    
                    drawdown = (peak - equity) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                metrics['max_drawdown'] = max_drawdown
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                'win_rate': 0.0,
                'profit_factor': 1.0,
                'total_return': 0.0,
                'max_drawdown': 0.0
            }
    
    def get_analysis_summary(self):
        """
        Get a summary of the price action analysis.
        
        Returns:
            dict: Summary of analysis results
        """
        analysis = self.analyze()
        signals = self.get_combined_signals()
        
        # Count signals
        buy_count = signals['buy'].sum()
        sell_count = signals['sell'].sum()
        
        # Calculate average signal strength
        avg_buy_strength = signals.loc[signals['buy'], 'buy_strength'].mean() if buy_count > 0 else 0
        avg_sell_strength = signals.loc[signals['sell'], 'sell_strength'].mean() if sell_count > 0 else 0
        
        # Determine overall bias with strength consideration
        if buy_count > sell_count and avg_buy_strength > avg_sell_strength:
            bias = 'strongly bullish'
        elif buy_count > sell_count:
            bias = 'bullish'
        elif sell_count > buy_count and avg_sell_strength > avg_buy_strength:
            bias = 'strongly bearish'
        elif sell_count > buy_count:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        # Get current market structure
        trend = analysis.get('market_structure', {}).get('trend', 'unknown')
        
        # Get key levels
        support_levels = analysis.get('market_structure', {}).get('support_levels', [])
        resistance_levels = analysis.get('market_structure', {}).get('resistance_levels', [])
        
        # Add price cluster support/resistance zones if available
        if 'pattern_recognition' in analysis:
            pr_clusters = analysis['pattern_recognition'].get('price_clusters', {})
            if pr_clusters:
                support_levels.extend(pr_clusters.get('support', []))
                resistance_levels.extend(pr_clusters.get('resistance', []))
                
                # Remove duplicates and sort
                support_levels = sorted(list(set(support_levels)))
                resistance_levels = sorted(list(set(resistance_levels)))
        
        # Get recent patterns
        recent_patterns = {}
        if 'candlestick' in analysis:
            patterns = analysis['candlestick']['patterns']
            for pattern_name, pattern_series in patterns.items():
                if pattern_series.iloc[-10:].any():
                    recent_patterns[pattern_name] = pattern_series.iloc[-10:].sum()
        
        # Get advanced patterns
        advanced_patterns = {}
        if 'pattern_recognition' in analysis:
            pr_data = analysis['pattern_recognition']
            
            # Count recent chart patterns
            advanced_patterns['channels'] = len(pr_data.get('channels', []))
            advanced_patterns['wedges'] = len(pr_data.get('wedges', []))
            
            # Handle double patterns safely
            double_patterns = pr_data.get('double_patterns', {})
            advanced_patterns['double_tops'] = len(double_patterns.get('double_top', []))
            advanced_patterns['double_bottoms'] = len(double_patterns.get('double_bottom', []))
            
            advanced_patterns['zigzags'] = len(pr_data.get('zigzag_patterns', []))
        
        # Get volume insights
        volume_insights = {}
        if 'volume' in analysis:
            volume_data = analysis['volume']
            volume_insights['recent_spikes'] = volume_data['volume_spikes'].iloc[-10:].sum()
            volume_insights['recent_bullish_divergence'] = volume_data['bullish_divergence'].iloc[-10:].sum()
            volume_insights['recent_bearish_divergence'] = volume_data['bearish_divergence'].iloc[-10:].sum()
        
        # Calculate performance metrics for price action signals
        performance_metrics = self._calculate_performance_metrics(signals)
        
        summary = {
            'bias': bias,
            'trend': trend,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'recent_patterns': recent_patterns,
            'advanced_patterns': advanced_patterns,
            'volume_insights': volume_insights,
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'avg_buy_strength': float(avg_buy_strength) if not pd.isna(avg_buy_strength) else 0,
            'avg_sell_strength': float(avg_sell_strength) if not pd.isna(avg_sell_strength) else 0,
            'win_rate': performance_metrics['win_rate'],
            'profit_factor': performance_metrics['profit_factor'],
            'total_return': performance_metrics['total_return'],
            'max_drawdown': performance_metrics['max_drawdown']
        }
        
        logger.info(f"Generated price action analysis summary with {bias} bias")
        return summary 