import ccxt
import pandas as pd
import datetime as dt
import sys
import time

sys.path.append("..")
import credential as cd

tickers = ["BTC", "ETH", "XRP", "DOGE", "SOL", "ADA", "BNB", "TRX"]
candle_columns = ["open", "high", "low", "close", "volume"]

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
    return pd.DataFrame(
        [kline[1:6] for kline in klines],
        index=pd.Index(
            [
                dt.datetime.fromtimestamp(kline[0] / 1000, dt.timezone.utc)
                for kline in klines
            ],
            name="time",
        ),
        columns=candle_columns,
    )


def main():

    price_map = {}
    for col in candle_columns:
        price_map[col] = {}

    for ticker in tickers:
        df_ohlcv = get_ohlc(ticker + "/USDT")
        for col in candle_columns:
            price_map[col][ticker] = df_ohlcv[col].rename(ticker)

    for col in candle_columns:
        df = pd.concat(price_map[col].values(), axis=1, join="outer")
        df.to_csv(f"{col}.csv")

        if col == "close":
            df_returns = df / df.shift(1) - 1
            df_returns.to_csv("returns.csv")


if __name__ == "__main__":
    main()
