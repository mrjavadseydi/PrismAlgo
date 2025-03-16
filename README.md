# Cryptocurrency Harmonic Pattern Analyzer

This application analyzes cryptocurrency data to identify which harmonic pattern algorithm works best with a given cryptocurrency. It fetches historical data, identifies harmonic patterns, and evaluates their performance through backtesting.

## Features

- Fetch cryptocurrency data from various exchanges including Binance.US
- Detect multiple harmonic patterns (Gartley, Butterfly, Bat, Crab, Shark, Cypher)
- Analyze price action patterns and market structure
- Backtest pattern performance with customizable parameters
- Generate performance reports and visualizations
- Compare different patterns to find the best performer

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. API keys (optional for public data):
   - For Binance.US: API keys are optional for fetching public market data
   - For Alpha Vantage: Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key) to get a free API key
   - Add your API key(s) to the `config.yaml` file or provide it via command line

## Configuration

The application uses a YAML configuration file (`config.yaml`) to specify the analysis parameters:

```yaml
# Exchange and symbol settings
exchange: 'binanceus'  # Binance US version - no regional restrictions
symbol: 'ETH/USD'      # Format: BASE/QUOTE

# API key settings (optional for public data)
# api_keys:  # Uncomment and add keys if needed for private API access
#   - 'YOUR_BINANCE_US_API_KEY'

# Time frame settings
timeframe: '4h'  # Options: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w

# Backtest settings
backtest:
  start_date: '2024-01-01'  # Format: YYYY-MM-DD
  end_date: ''              # Empty for current date
  
# Harmonic patterns to analyze
patterns:
  - 'gartley'
  - 'butterfly'
  - 'bat'
  - 'crab'
  - 'shark'
  - 'cypher'

# Trading parameters
trading:
  initial_capital: 10000
  position_size: 0.1  # Percentage of capital per trade
  stop_loss: 0.02     # Percentage from entry
  take_profit: 0.05   # Percentage from entry

# Output settings
output:
  save_results: true
  plot_charts: true
  verbose: true

# Price action settings
price_action:
  enabled: true
  detect_support_resistance: true
  analyze_candlesticks: true
  volume_analysis: true
```

## Usage

Run the application with the default configuration:

```
python app.py
```

Or specify a custom configuration file:

```
python app.py --config my_config.yaml
```

You can also specify a custom output directory:

```
python app.py --output-dir my_results
```

To provide an API key via command line (overrides the config file):

```
python app.py --api-key YOUR_API_KEY
```

## Supported Data Sources

The application supports multiple data sources:

1. **Binance.US** (Recommended)
   - No regional restrictions in the US
   - High rate limits and good data availability
   - No API key required for public market data
   - Set `exchange: 'binanceus'` in the config file

2. **Alpha Vantage** (Free API with strict limits)
   - Provides free cryptocurrency data with rate limits (5 calls/minute, 500/day)
   - Requires an API key (get one at https://www.alphavantage.co/support/#api-key)
   - Set `exchange: 'alphavantage'` in the config file
   - Limited cryptocurrency symbols

3. **Other CCXT-supported exchanges**
   - Supports numerous cryptocurrency exchanges through the CCXT library
   - Some exchanges may require API keys for data access
   - Example: `exchange: 'coinbase'` or `exchange: 'kraken'`
   - Note: Regular Binance (`exchange: 'binance'`) has regional restrictions

## Output

The application generates the following outputs:

1. Summary report of pattern performance
2. Detailed trade results for each pattern
3. Equity curve plots
4. Pattern visualization charts
5. Log file with detailed execution information

All outputs are saved in the specified output directory (default: `results/`).

## Supported Harmonic Patterns

The application supports the following harmonic patterns:

- Gartley
- Butterfly
- Bat
- Crab
- Shark
- Cypher

Each pattern has specific Fibonacci ratio requirements and is detected based on price swing points.

## Price Action Algorithms

The application also includes price action analysis algorithms that can be used alongside harmonic patterns:

### Market Structure Analysis

- **Support and Resistance Detection**: Identifies key price levels where the market has historically reversed
- **Trend Analysis**: Determines the current market trend using higher highs/higher lows (uptrend) or lower highs/lower lows (downtrend)
- **Swing Point Detection**: Identifies significant swing highs and swing lows in the price action

### Candlestick Patterns

- **Engulfing Patterns**: Detects bullish and bearish engulfing patterns
- **Doji Formations**: Identifies doji candlesticks that signal potential reversals
- **Pin Bars**: Detects pin bar (hammer/shooting star) formations that indicate rejection of price levels

### Volume Analysis

- **Volume Confirmation**: Analyzes volume to confirm the strength of price movements
- **Volume Divergence**: Detects divergences between price action and volume

### Combining Approaches

For optimal results, the application can combine harmonic patterns with price action analysis:
- Use price action to confirm harmonic pattern completions
- Filter harmonic pattern signals based on market structure
- Enhance entry and exit timing using candlestick patterns

## Extending the Application

### Adding New Patterns

To add a new harmonic pattern:

1. Create a new pattern class in the `src/patterns/` directory
2. Inherit from the `BaseHarmonicPattern` class
3. Implement the required methods
4. Add the pattern to the `PatternFactory` class

### Supporting New Data Sources

To add support for a new data source:

1. Extend the `DataFetcher` class in `src/data/fetcher.py`
2. Implement the required methods for fetching and processing data
3. Update the configuration handling to support the new data source

## License

This project is licensed under the MIT License - see the LICENSE file for details.
