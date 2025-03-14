import yaml
import logging
import os
from datetime import datetime

logger = logging.getLogger('Config')

class Config:
    """
    Configuration handler for the application.
    """
    
    def __init__(self, config_file='config.yaml'):
        """
        Initialize the configuration handler.
        
        Args:
            config_file (str): Path to the configuration file
        """
        self.config_file = config_file
        self.config = None
        
        self.load_config()
        self.validate_config()
    
    def load_config(self):
        """
        Load configuration from the YAML file.
        """
        try:
            with open(self.config_file, 'r') as file:
                self.config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def validate_config(self):
        """
        Validate the configuration.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        # Check if required sections exist
        required_sections = ['exchange', 'symbol', 'timeframe', 'backtest']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate exchange
        exchange = self.config['exchange'].lower()
        
        # Check for API keys when using Alpha Vantage
        if exchange == 'alphavantage':
            if 'api_keys' not in self.config or not self.config['api_keys']:
                logger.warning("No API keys provided for Alpha Vantage. Get free keys at https://www.alphavantage.co/support/#api-key")
            elif isinstance(self.config['api_keys'], list):
                logger.info(f"Found {len(self.config['api_keys'])} API keys for Alpha Vantage")
                # Check for default/demo keys
                for key in self.config['api_keys']:
                    if key in ['DEMO', 'demo', 'YOUR_API_KEY']:
                        logger.warning(f"Found default/demo API key: {key}. This may have limited functionality.")
            else:
                # Convert single key to list
                self.config['api_keys'] = [self.config['api_keys']]
                logger.info("Converted single API key to list format")
        
        # For backward compatibility, check for api_key (singular)
        if 'api_key' in self.config and 'api_keys' not in self.config:
            logger.info("Converting legacy 'api_key' to 'api_keys' list format")
            self.config['api_keys'] = [self.config['api_key']]
        
        # Validate symbol format
        if '/' not in self.config['symbol']:
            logger.warning(f"Symbol format may be incorrect: {self.config['symbol']}. Recommended format: BASE/QUOTE (e.g., BTC/USD)")
        
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '1d', '1w', '1M']
        if self.config['timeframe'] not in valid_timeframes:
            logger.warning(f"Timeframe {self.config['timeframe']} may not be supported by all exchanges. Valid timeframes: {', '.join(valid_timeframes)}")
        
        # Validate backtest dates
        try:
            start_date = datetime.strptime(self.config['backtest']['start_date'], '%Y-%m-%d')
            if 'end_date' in self.config['backtest'] and self.config['backtest']['end_date']:
                end_date = datetime.strptime(self.config['backtest']['end_date'], '%Y-%m-%d')
                if end_date < start_date:
                    raise ValueError("End date cannot be earlier than start date")
                
                # Check if dates are in the future
                now = datetime.now()
                if start_date > now or end_date > now:
                    logger.warning("Backtest dates include future dates. This may result in empty data.")
        except ValueError as e:
            raise ValueError(f"Invalid date format in backtest configuration: {e}")
        
        logger.info("Configuration validated successfully")
    
    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key (str): Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        if '.' in key:
            # Handle nested keys
            parts = key.split('.')
            value = self.config
            for part in parts:
                if part not in value:
                    return default
                value = value[part]
            return value
        
        return self.config.get(key, default)
    
    def get_patterns(self):
        """
        Get the list of patterns to analyze.
        
        Returns:
            list: List of pattern names
        """
        return self.config.get('patterns', [])
    
    def get_trading_params(self):
        """
        Get trading parameters.
        
        Returns:
            dict: Dictionary of trading parameters
        """
        trading = self.config.get('trading', {})
        return {
            'initial_capital': trading.get('initial_capital', 10000),
            'position_size': trading.get('position_size', 0.1),
            'stop_loss': trading.get('stop_loss', 0.02),
            'take_profit': trading.get('take_profit', 0.05)
        }
    
    def get_output_params(self):
        """
        Get output parameters.
        
        Returns:
            dict: Dictionary of output parameters
        """
        output = self.config.get('output', {})
        return {
            'save_results': output.get('save_results', True),
            'plot_charts': output.get('plot_charts', True),
            'verbose': output.get('verbose', True)
        } 