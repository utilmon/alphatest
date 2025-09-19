# import schwabdev # couldn't generate tokens
import credential as cd
from schwab import auth, client  # https://github.com/alexgolec/schwab-py
import json
import pandas as pd
import datetime as dt
import numpy as np
from pytz import timezone

#### schwab
tickers = ["VOO", "GLD"]
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


for ticker in tickers:
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


def get_ohlc(symbol: str, interval: str = "1d") -> pd.DataFrame:
    klines = bn.fetch_ohlcv(symbol, interval)
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

df_ohlcv = get_ohlc("BTC/USDT")
df_returns = df_ohlcv.close / df_ohlcv.close.shift(1) - 1
returns_map["BTC"] = df_returns.rename("BTC")

####


df = pd.concat(returns_map.values(), axis=1, join="outer")
df.to_csv("returns.csv")
print(df)
