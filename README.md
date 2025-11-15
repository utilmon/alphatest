<div align="center">
  <img src="https://github.com/utilmon/alphatest/blob/main/docs/logo.png?raw=true" alt="AlphaTest Logo" width="250">
</div>

# **AlphaTest: Quantitative Strategy Backtester**

<p align="center">
  <a href="https://pypi.org/project/alphatest/"><img src="https://img.shields.io/pypi/v/alphatest.svg" alt="PyPI version" /></a>&nbsp;<a href="https://pypi.org/project/alphatest/"><img src="https://img.shields.io/pypi/pyversions/alphatest.svg" alt="Python versions" /></a>&nbsp;<a href="https://github.com/utilmon/alphatest/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/alphatest.svg" alt="License" /></a>
</p>

**AlphaTest** is a lightweight, vectorized Python framework designed for backtesting quantitative trading strategies. It supports multi-asset portfolios, realistic transaction cost modeling, and benchmarking against both Equity and Cryptocurrency indices.  
It is designed to ingest alpha signals (DataFrames of position weights or raw signals) and compute rigorous performance metrics and visualizations.

## **Key Features**

- **Vectorized Performance Analysis:** Leverages pandas and numpy for fast, element-wise operations on large datasets.
- **Hybrid Benchmarking:** Built-in support for comparing strategies against:
  - **Equities:** S\&P 500 (VOO), Gold (GLD) — (Annualized: 252 days).
  - **Crypto:** Bitcoin (BTC) — (Annualized: 365 days).
- **Comprehensive Risk Metrics:** Automatically computes Sharpe, Sortino, Calmar, Maximum Drawdown, CAGR, Profit Factor, and Turnover.
- **Cost Modeling:** Simulates trading fees based on daily turnover to produce net-of-fee returns.
- **Visualization:** Generates dual-panel plots showing Log-Scale Cumulative Returns and Rolling Drawdowns.
- **Signal Distribution:** Helper methods to analyze the statistical distribution of alpha signals using the Freedman-Diaconis rule for optimal binning.

## **Dependencies**

- Python 3.x
- pandas
- numpy
- matplotlib

## **Directory Structure & Data Requirements**

The backtest class relies on a specific file system structure to locate your asset data and benchmark data.

### **File System Layout**

```Plaintext

project_root/
│
├── src/
│   ├── alphatest.py         \# This package
│   ├── util.py              \# Must contain 'cross\_sectional' class (imported as cs)
│
├── data/                \# Your primary data\_path
│   ├── returns.csv      \# Required: Asset returns data
│   ├── benchmarks/          \# Hardcoded benchmark path (relative to data\_path parent)
│   │      └── returns.csv      \# Must contain columns: 'BTC', 'VOO', 'GLD'
│   └── ...              \# Other feature files (e.g. volume.csv)
│
```

### **Data Format (.csv)**

Input files must contain a time column to be parsed as the index.

```
time,Asset_A,Asset_B,Asset_C
2023-01-01,0.01,-0.02,0.005
2023-01-02,0.005,0.01,-0.01
...
```

## **Usage**

### **1\. Initialization**

Initialize the backtester by pointing it to your data directory. You can customize risk-free rates, trading fees, and trading frequency.

```Python

from alphatest import backtest
import pandas as pd

# Initialize the engine
bt = backtest(
    data_path="./data",
    annual_days=365,          \# 365 for Crypto, 252 for TradFi
    risk_free_return=0.03,    \# 3% Risk Free Rate
    trading_fee=0.00025       \# 2.5 bps taker fee
)
```

### **2\. Running a Backtest**

Pass a DataFrame of alpha signals to the run method.

- **Input:** A DataFrame where columns are assets and index is time.
- **Scaling:** By default, run will neutralize and scale your signals (using scale_final=True).

```Python

# Load or generate your alpha dataframe
# Shape: (Time, Assets)
alpha_df = pd.read_csv("my_signals.csv", index_col=0, parse_dates=True)

# Execute backtest
metrics = bt.run(alpha_df, scale_final=True)

# 'metrics' is a dictionary containing the results
print(f"Strategy Sharpe: {metrics['sharpe']:.2f}")
```

### **3\. Visualizing Alpha Distribution**

Check the statistical properties of your raw signal before backtesting.

```Python

bt.get_distribution(alpha_df)
```

## **Metrics Glossary**

The compute_metrics method calculates the following statistics:

| Metric                   | Description                                            |
| :----------------------- | :----------------------------------------------------- |
| **CAGR (Annual Return)** | Geometric annual return compounded over the period.    |
| **Annual Volatility**    | Standard deviation of returns \* sqrt(annual_ticks).   |
| **Sharpe Ratio**         | (CAGR \- RiskFree) / Annual Volatility.                |
| **Sortino Ratio**        | (CAGR \- RiskFree) / Downside Deviation.               |
| **Calmar Ratio**         | (CAGR \- RiskFree) / abs(Max Drawdown).                |
| **Max Drawdown**         | The largest peak-to-trough decline in portfolio value. |
| **Profit Factor**        | Gross Profits / Gross Losses.                          |
| **Turnover**             | Mean daily portfolio turnover (percentage).            |

## **Configuration Options**

| Parameter         | Default          | Description                                      |
| :---------------- | :--------------- | :----------------------------------------------- |
| annual_days       | 365              | Trading days per year (Use 252 for Stocks).      |
| daily_ticks       | 1                | Observations per day (e.g., 24 for hourly).      |
| risk_free_return  | 0.03             | Annualized risk-free rate (decimal).             |
| trading_fee       | 0.00025          | Transaction cost per trade (decimal).            |
| crypto_benchmarks | \[``BTC"\]       | List of crypto ticker columns in benchmark file. |
| stock_benchmarks  | \[`VOO", `GLD"\] | List of stock ticker columns in benchmark file.  |

## **Notes**

- **Scaling:** The engine assumes the presence of a .util module with a cross_sectional class for signal neutralization/scaling.
- **Returns:** The engine uses **Geometric Compounding** for cumulative returns and **Simple Returns** for daily input data.
