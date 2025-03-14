# Cryptocurrency Harmonic Pattern Analyzer

This application analyzes cryptocurrency data to identify which harmonic pattern algorithm works best with a given cryptocurrency. It fetches historical data, identifies harmonic patterns, and evaluates their performance through backtesting.

## Features

- Fetch cryptocurrency data from various exchanges or free APIs like Alpha Vantage
- Detect multiple harmonic patterns (Gartley, Butterfly, etc.)
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

3. Get a free API key:
   - For Alpha Vantage: Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key) to get a free API key
   - Add your API key to the `config.yaml` file or provide it via command line

## Configuration

The application uses a YAML configuration file (`config.yaml`) to specify the analysis parameters:

```yaml
# Exchange and symbol settings
exchange: 'alphavantage'  # Free API alternative to paid exchanges
symbol: 'BTC/USD'  # Format: BASE/QUOTE

# API key settings
api_key: 'YOUR_ALPHA_VANTAGE_API_KEY'  # Get a free key at https://www.alphavantage.co/support/#api-key

# Time frame settings
timeframe: '1d'  # Options: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w

# Backtest settings
backtest:
  start_date: '2023-01-01'  # Format: YYYY-MM-DD
  end_date: '2023-12-31'    # Format: YYYY-MM-DD
  
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

1. **Alpha Vantage** (Free API with limits)
   - Provides free cryptocurrency data with a rate limit
   - Requires an API key (get one at https://www.alphavantage.co/support/#api-key)
   - Set `exchange: 'alphavantage'` in the config file

2. **CCXT-supported exchanges**
   - Supports numerous cryptocurrency exchanges through the CCXT library
   - Some exchanges may require API keys for data access
   - Example: `exchange: 'binance'` or `exchange: 'coinbase'`

## Output

The application generates the following outputs:

1. Summary report of pattern performance
2. Detailed trade results for each pattern
3. Equity curve plots
4. Pattern visualization charts
5. Log file with detailed execution information

All outputs are saved in the specified output directory (default: `results/`).

## Supported Harmonic Patterns

Currently, the application supports the following harmonic patterns:

- Gartley
- Butterfly

More patterns can be added by implementing additional pattern classes.

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

This project is licensed under the MIT License - see the LICENSE file for details. # PrismAlgo
