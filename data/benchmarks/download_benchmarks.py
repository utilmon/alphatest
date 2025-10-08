# import schwabdev # couldn't generate tokens
import credential as cd
from schwab import auth, client  # https://github.com/alexgolec/schwab-py
import json
import pandas as pd
import datetime as dt
import numpy as np
from pytz import timezone
import time

###

stock_tickers = ["VOO", "GLD", "TLT", "QQQ"]
crypto_tickers = ["BTC", "ETH"]


#### schwab
returns_map = {}

api_key = cd.app_key
app_secret = cd.secret
callback_url = "https://127.0.0.1"
token_path = "./token.json"

c = auth.easy_client(api_key, app_secret, callback_url, token_path)  # for script


def get_ohlc(symbol: str) -> pd.DataFrame:
    candles = c.get_price_history_every_day(symbol).json()["candles"]
    datetime = [candle["datetime"] for candle in candles]
    data = []
    columns = ["open", "high", "low", "close", "volume"]
    for col in columns:
        data.append([candle[col] for candle in candles])
    data = np.transpose(data)
    return pd.DataFrame(
        data,
        index=pd.Index(
            [
                dt.datetime.fromtimestamp(
                    datetime_data / 1000, timezone("America/New_York")
                )
                for datetime_data in datetime
            ],
            name="time",
        ),
        columns=columns,
    )


for ticker in stock_tickers:
    df_ohlcv = get_ohlc(ticker)
    df_returns = df_ohlcv.close / df_ohlcv.close.shift(1) - 1
    returns_map[ticker] = df_returns.rename(ticker)

##### Binance
import ccxt

bn = ccxt.binance({"apiKey": cd.api_key, "secret": cd.api_secret})
bn.options = {
    "defaultType": "margin",
    "defaultMarginMode": "cross",
    "adjustForTimeDifference": True,
    "newOrderRespType": "FULL",
    "defaultTimeInForce": "GTC",
    "enableRateLimit": True,
    "rate_limit": 1100,  # (ms)
}


def fetch_historical_ohlcv(symbol, timeframe="1d", verbose=True):
    """
    Fetches historical OHLCV data from Binance in iterative chunks.

    :param symbol: str, the trading pair symbol (e.g., 'BTC/USDT').
    :param timeframe: str, the timeframe for the candles (e.g., '1d', '4h', '1m').
    :param since_date: str, the start date in 'YYYY-MM-DD' format.
    :return: list, a list of lists containing OHLCV data.
    """

    # Convert the start date string to a milliseconds timestamp
    # since_timestamp = bn.parse_date(f'{since_date}T00:00:00Z')
    since_timestamp = 0

    # List to hold all the fetched OHLCV data
    all_ohlcv = []

    # The API limit per request
    limit = 500

    if verbose:
        print(f"Starting to fetch {symbol} {timeframe} data from {0}...")

    while True:
        try:
            # Fetch a chunk of OHLCV data
            # The 'since' parameter is the starting timestamp in milliseconds
            # The 'limit' parameter is the number of candles to fetch
            ohlcv_chunk = bn.fetch_ohlcv(
                symbol, timeframe, since=since_timestamp, limit=limit
            )

            # If the fetch returned no data, we've reached the end
            if not ohlcv_chunk:
                if verbose:
                    print("No more data available. Fetch complete.")
                break

            # Get the timestamp of the last candle in the chunk
            last_timestamp = ohlcv_chunk[-1][0]

            # Append the fetched chunk to our main list
            all_ohlcv.extend(ohlcv_chunk)

            if verbose:
                print(
                    f"Fetched {len(ohlcv_chunk)} candles. "
                    f"From: {dt.datetime.fromtimestamp(ohlcv_chunk[0][0] / 1000).strftime('%Y-%m-%d %H:%M:%S')} "
                    f"To: {dt.datetime.fromtimestamp(last_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')}"
                )

            # If the number of candles returned is less than the limit,
            # it means we have fetched all the data up to the present
            if len(ohlcv_chunk) < limit:
                if verbose:
                    print("Fetched the last available chunk of data. Fetch complete.")
                break

            # Set the 'since' for the next iteration to be the timestamp of the last candle + 1 millisecond
            # to avoid fetching the same candle again.
            since_timestamp = last_timestamp + 1

            # The ccxt library has built-in rate limiting, but a small sleep can sometimes help
            # especially if you are making other API calls. The `enableRateLimit` above handles this automatically.
            # time.sleep(exchange.rateLimit / 1000) # This is generally handled by the exchange instance

        except ccxt.NetworkError as e:
            print(f"Network error: {e}. Retrying in 30 seconds...")
            time.sleep(30)
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {e}. Retrying in 60 seconds...")
            time.sleep(60)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

    return all_ohlcv


def get_ohlc(symbol: str, interval: str = "1d") -> pd.DataFrame:
    klines = fetch_historical_ohlcv(symbol, interval, verbose=False)
    columns = ["open", "high", "low", "close", "volume"]
    return pd.DataFrame(
        [kline[1:6] for kline in klines],
        index=pd.Index(
            [
                dt.datetime.fromtimestamp(kline[0] / 1000, dt.timezone.utc)
                for kline in klines
            ],
            name="time",
        ),
        columns=columns,
    )

for ticker in crypto_tickers:
    df_ohlcv = get_ohlc(ticker + "/USDT")
    df_returns = df_ohlcv.close / df_ohlcv.close.shift(1) - 1
    returns_map[ticker] = df_returns.rename(ticker)

####


df = pd.concat(returns_map.values(), axis=1, join="outer")
df.to_csv("returns.csv")
print(df)
