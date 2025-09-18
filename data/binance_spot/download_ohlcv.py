import ccxt
import pandas as pd
import datetime as dt
import sys

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


def get_ohlc(symbol: str, interval: str = "1d") -> pd.DataFrame:
    klines = bn.fetch_ohlcv(symbol, interval)
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
