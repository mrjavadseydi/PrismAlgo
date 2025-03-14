import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import requests
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataFetcher')

class DataFetcher:
    """
    Class for fetching cryptocurrency data from various exchanges.
    """
    
    def __init__(self, exchange_name, symbol, timeframe, api_key=None):
        """
        Initialize the DataFetcher with exchange and symbol information.
        
        Args:
            exchange_name (str): Name of the exchange (e.g., 'binance', 'coinbase', 'alphavantage')
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            timeframe (str): Candle timeframe (e.g., '1h', '1d')
            api_key (str or list, optional): API key(s) for services that require it (e.g., Alpha Vantage)
                                            Can be a single key or a list of keys
        """
        self.exchange_name = exchange_name.lower()
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Handle multiple API keys
        if isinstance(api_key, list):
            self.api_keys = api_key
            logger.info(f"Initialized with {len(self.api_keys)} API keys")
        else:
            self.api_keys = [api_key] if api_key else []
            
        # Initialize exchange or API client
        if self.exchange_name == 'alphavantage':
            if not self.api_keys:
                logger.warning("Alpha Vantage API key not provided. Get a free key at https://www.alphavantage.co/support/#api-key")
            self.base_url = "https://www.alphavantage.co/query"
            # Parse symbol for Alpha Vantage format (e.g., BTC/USDT -> BTC)
            self.base_currency = symbol.split('/')[0]
            self.quote_currency = symbol.split('/')[1] if '/' in symbol else 'USD'
            logger.info(f"Initialized Alpha Vantage API client for {self.base_currency}/{self.quote_currency}")
        else:
            try:
                exchange_class = getattr(ccxt, exchange_name)
                self.exchange = exchange_class({
                    'enableRateLimit': True,  # Required by most exchanges
                    'apiKey': self.api_keys[0] if self.api_keys else None
                })
                logger.info(f"Successfully initialized {exchange_name} exchange")
            except Exception as e:
                logger.error(f"Failed to initialize exchange {exchange_name}: {e}")
                raise
    
    def _get_random_api_key(self):
        """
        Get a random API key from the available keys.
        
        Returns:
            str: A randomly selected API key or None if no keys are available
        """
        if not self.api_keys:
            logger.warning("No API keys available")
            return None
        
        selected_key = random.choice(self.api_keys)
        logger.info(f"Using API key: {selected_key[:5]}...{selected_key[-5:] if len(selected_key) > 10 else ''}")
        return selected_key
    
    def _map_timeframe_to_alphavantage(self, timeframe):
        """
        Map CCXT timeframe to Alpha Vantage interval.
        
        Args:
            timeframe (str): CCXT timeframe format
            
        Returns:
            str: Alpha Vantage interval
        """
        mapping = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '60min',
            '1d': 'daily',
            '1w': 'weekly',
            '1M': 'monthly'
        }
        
        if timeframe not in mapping:
            raise ValueError(f"Timeframe {timeframe} not supported by Alpha Vantage. Supported timeframes: {', '.join(mapping.keys())}")
        
        return mapping[timeframe]
    
    def fetch_alphavantage_data(self, start_date, end_date=None):
        """
        Fetch cryptocurrency data from Alpha Vantage API.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to current date.
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        api_key = self._get_random_api_key()
        if not api_key:
            raise ValueError("Alpha Vantage API key is required. Get a free key at https://www.alphavantage.co/support/#api-key")
        
        # Map timeframe to Alpha Vantage interval
        interval = self._map_timeframe_to_alphavantage(self.timeframe)
        
        # Determine which API function to use based on interval
        if interval in ['1min', '5min', '15min', '30min', '60min']:
            function = 'CRYPTO_INTRADAY'
            params = {
                'function': function,
                'symbol': self.base_currency,
                'market': self.quote_currency,
                'interval': interval,
                'apikey': api_key,
                'outputsize': 'full'
            }
        else:
            function = 'DIGITAL_CURRENCY_' + interval.upper()
            params = {
                'function': function,
                'symbol': self.base_currency,
                'market': self.quote_currency,
                'apikey': api_key
            }
        
        logger.info(f"Fetching {self.base_currency}/{self.quote_currency} data from Alpha Vantage with interval {interval}")
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            
            # Check for error messages
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
            
            if 'Information' in data:
                logger.info(f"Alpha Vantage info: {data['Information']}")
                # Check if we hit API call limits
                if "call frequency" in data['Information'].lower():
                    raise ValueError(f"Alpha Vantage API limit reached: {data['Information']}")
            
            # Parse the data based on the function used
            if function == 'CRYPTO_INTRADAY':
                time_series_key = f"Time Series Crypto ({interval})"
                if time_series_key not in data:
                    raise ValueError(f"Expected key '{time_series_key}' not found in Alpha Vantage response")
                
                time_series = data[time_series_key]
                ohlcv_data = []
                
                for timestamp, values in time_series.items():
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    ohlcv_data.append([
                        dt,
                        float(values['1. open']),
                        float(values['2. high']),
                        float(values['3. low']),
                        float(values['4. close']),
                        float(values['5. volume'])
                    ])
            else:
                # For daily, weekly, monthly data
                # First, find the correct time series key by checking available keys
                time_series_key = None
                for key in data.keys():
                    if key.startswith("Time Series") or key.startswith("Digital Currency"):
                        time_series_key = key
                        break
                
                if not time_series_key:
                    raise ValueError(f"No time series data found in Alpha Vantage response. Available keys: {', '.join(data.keys())}")
                
                logger.info(f"Found time series key: {time_series_key}")
                time_series = data[time_series_key]
                ohlcv_data = []
                
                # Inspect the first entry to determine the format
                if time_series:
                    first_timestamp = next(iter(time_series))
                    first_values = time_series[first_timestamp]
                    
                    # Log the available keys for debugging
                    logger.info(f"Available data keys: {', '.join(first_values.keys())}")
                    
                    # Try to determine the correct keys for OHLCV data
                    open_key = None
                    high_key = None
                    low_key = None
                    close_key = None
                    volume_key = None
                    
                    # Check for different possible formats
                    for key in first_values.keys():
                        if 'open' in key.lower():
                            open_key = key
                        elif 'high' in key.lower():
                            high_key = key
                        elif 'low' in key.lower():
                            low_key = key
                        elif 'close' in key.lower():
                            close_key = key
                        elif 'volume' in key.lower():
                            volume_key = key
                    
                    # If we couldn't find all required keys, try standard patterns
                    if not all([open_key, high_key, low_key, close_key, volume_key]):
                        # Try standard format first
                        if '1. open' in first_values:
                            open_key = '1. open'
                            high_key = '2. high'
                            low_key = '3. low'
                            close_key = '4. close'
                            volume_key = '5. volume'
                        # Try digital currency format
                        elif f'1a. open ({self.quote_currency})' in first_values:
                            open_key = f'1a. open ({self.quote_currency})'
                            high_key = f'2a. high ({self.quote_currency})'
                            low_key = f'3a. low ({self.quote_currency})'
                            close_key = f'4a. close ({self.quote_currency})'
                            volume_key = '5. volume'
                        # Try another common format
                        elif f'1b. open ({self.quote_currency})' in first_values:
                            open_key = f'1b. open ({self.quote_currency})'
                            high_key = f'2b. high ({self.quote_currency})'
                            low_key = f'3b. low ({self.quote_currency})'
                            close_key = f'4b. close ({self.quote_currency})'
                            volume_key = '5. volume'
                    
                    if not all([open_key, high_key, low_key, close_key, volume_key]):
                        # If we still can't find the keys, try a fallback approach
                        # Use the first keys that contain 'open', 'high', 'low', 'close', and 'volume'
                        for key in first_values.keys():
                            if 'open' in key.lower() and not open_key:
                                open_key = key
                            elif 'high' in key.lower() and not high_key:
                                high_key = key
                            elif 'low' in key.lower() and not low_key:
                                low_key = key
                            elif 'close' in key.lower() and not close_key:
                                close_key = key
                            elif 'volume' in key.lower() and not volume_key:
                                volume_key = key
                    
                    if not all([open_key, high_key, low_key, close_key, volume_key]):
                        raise ValueError(f"Could not determine OHLCV keys from Alpha Vantage response. Available keys: {', '.join(first_values.keys())}")
                    
                    logger.info(f"Using OHLCV keys: {open_key}, {high_key}, {low_key}, {close_key}, {volume_key}")
                    
                    # Parse the data with the determined keys
                    for timestamp, values in time_series.items():
                        try:
                            dt = datetime.strptime(timestamp, '%Y-%m-%d')
                            ohlcv_data.append([
                                dt,
                                float(values[open_key]),
                                float(values[high_key]),
                                float(values[low_key]),
                                float(values[close_key]),
                                float(values[volume_key])
                            ])
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Error parsing data for {timestamp}: {e}")
                            continue
                else:
                    raise ValueError("No time series data found in Alpha Vantage response")
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Filter by date range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            if end_date:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            else:
                df = df[df.index >= start_dt]
            
            logger.info(f"Successfully fetched {len(df)} data points from Alpha Vantage")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Alpha Vantage: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage data: {e}")
            return pd.DataFrame()
    
    def fetch_ohlcv(self, start_date, end_date=None):
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for the specified symbol and timeframe.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to current date.
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        # Use Alpha Vantage API if selected
        if self.exchange_name == 'alphavantage':
            return self.fetch_alphavantage_data(start_date, end_date)
        
        # Otherwise use CCXT
        if not self.exchange.has['fetchOHLCV']:
            raise Exception(f"{self.exchange_name} does not support fetching OHLCV data")
        
        # Convert dates to timestamps
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        if end_date:
            end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            end_timestamp = int(datetime.now().timestamp() * 1000)
        
        logger.info(f"Fetching {self.symbol} data from {start_date} to {end_date or 'now'}")
        
        # Fetch data in chunks to avoid rate limits
        all_candles = []
        current_timestamp = start_timestamp
        
        while current_timestamp < end_timestamp:
            try:
                logger.info(f"Fetching chunk starting from {datetime.fromtimestamp(current_timestamp/1000)}")
                candles = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=current_timestamp,
                    limit=1000  # Most exchanges limit to 1000 candles per request
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update timestamp for next iteration
                current_timestamp = candles[-1][0] + 1
                
                # Respect rate limits
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break
        
        if not all_candles:
            logger.warning("No data was fetched")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Filter by end date
        if end_date:
            df = df[df.index <= end_date]
        
        logger.info(f"Successfully fetched {len(df)} candles")
        return df
    
    def get_latest_price(self):
        """
        Get the latest price for the symbol.
        
        Returns:
            float: Latest price
        """
        if self.exchange_name == 'alphavantage':
            try:
                api_key = self._get_random_api_key()
                if not api_key:
                    raise ValueError("Alpha Vantage API key is required")
                    
                params = {
                    'function': 'CURRENCY_EXCHANGE_RATE',
                    'from_currency': self.base_currency,
                    'to_currency': self.quote_currency,
                    'apikey': api_key
                }
                
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'Realtime Currency Exchange Rate' in data:
                    return float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])
                else:
                    logger.error(f"Unexpected response format from Alpha Vantage: {data}")
                    return None
            except Exception as e:
                logger.error(f"Error fetching latest price from Alpha Vantage: {e}")
                return None
        else:
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                return ticker['last']
            except Exception as e:
                logger.error(f"Error fetching latest price: {e}")
                return None 