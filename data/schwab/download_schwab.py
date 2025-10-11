# import schwabdev # couldn't generate tokens
import sys
sys.path.append("..")
import credential as cd
from schwab import auth, client  # https://github.com/alexgolec/schwab-py
import pandas as pd
import datetime as dt
import numpy as np
from pytz import timezone

###

stock_tickers = ["VOO", "GLD", "TLT", "QQQ","QLD","UBT","IEF","DBC","UUP"]
candle_columns = ["open", "high", "low", "close", "volume"]

#### schwab

api_key = cd.app_key
app_secret = cd.secret
callback_url = "https://127.0.0.1"
token_path = "../token.json"

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

def main():

    price_map = {}
    for col in candle_columns:
        price_map[col] = {}

    for ticker in stock_tickers:
        df_ohlcv = get_ohlc(ticker)
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