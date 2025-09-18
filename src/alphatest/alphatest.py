import pandas as pd
from .util import cross_sectional as cs
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 6)

####

alpha_path = r"C:\Users\qwane\Documents\git\tvl_alpha\backtest\tvl_diff7.csv"
price_data_path = r"C:\Users\qwane\Documents\git\market_data\binance\spot"
exclude = ["cro", "op", "bera"]
trading_fee = 0.00045

####


class backtest:
    def __init__(self, annual_days=365, risk_free_return=0.03):
        self.annual_days = annual_days
        self.price_data_path = price_data_path
        self.fee = trading_fee
        self.risk_free_return = 0.03

    def csv_to_df(self, path: str):
        return pd.read_csv(path, index_col="date")

    def get_returns_df(self, tickers: list):
        returns_map = {}
        for ticker in tickers:
            price_df = pd.read_csv(
                os.path.join(self.price_data_path, f"{ticker}.csv"), index_col="date"
            )
            returns_sr = (price_df.close / price_df.close.shift(1)) - 1
            returns_map[ticker] = returns_sr.rename(ticker)
        return pd.concat(returns_map.values(), axis=1).fillna(0)

    def compute_turnover(self, alpha_df: pd.DataFrame):
        diff_df = alpha_df.diff()
        turnover = diff_df.abs().sum(axis=1)
        return turnover

    def compute_position(self, alpha_df: pd.DataFrame):
        return alpha_df.sum(axis=1).mean()

    def compute_metrics(self, strategy_return: pd.Series):
        total_days = strategy_return.shape[0]
        cumulative = (strategy_return + 1).prod()
        annual_return = (cumulative ** (self.annual_days / total_days)) - 1
        annual_vol = np.sqrt(self.annual_days) * strategy_return.std()
        sharpe = (
            (annual_return - self.risk_free_return) / annual_vol
            if annual_vol != 0
            else np.nan
        )
        downside_std = strategy_return[strategy_return < 0].std() * np.sqrt(
            self.annual_days
        )
        sortino = (
            (annual_return - self.risk_free_return) / downside_std
            if downside_std != 0
            else np.nan
        )
        cum_returns = (strategy_return + 1).cumprod()
        drawdown = cum_returns / cum_returns.cummax() - 1
        max_drawdown = drawdown.min()
        calmar = (
            (annual_return - self.risk_free_return) / -max_drawdown
            if max_drawdown != 0
            else np.nan
        )
        return [annual_return, annual_vol, sharpe, sortino, max_drawdown, calmar]

    def run(self, alpha_df: pd.DataFrame):

        alpha_df = alpha_df.apply(cs.scale_final, axis=1)
        returns_df = self.get_returns_df(alpha_df.columns.values)
        common_rows = alpha_df.shift(1).index.intersection(returns_df.index)
        strategy_return = (
            (returns_df.loc[common_rows, alpha_df.columns] * alpha_df.shift(1))
            .dropna()
            .sum(axis=1)
        )
        turnover = self.compute_turnover(alpha_df)
        strategy_return -= turnover * self.fee
        metrics = self.compute_metrics(strategy_return)
        annual_turnover = turnover.mean() * self.annual_days
        long_short = self.compute_position(alpha_df)
        metrics.extend([annual_turnover * 100, long_short])

        metrics_df = pd.DataFrame(
            {
                "Metrics": [
                    "CAGR",
                    "Annualized Volatility",
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Maximum Drawdown",
                    "Calmar Ratio",
                    "Annual Turnover (%)",
                    "Long/Short position",
                ],
                "Strategy": [f"{metric:.2f}" for metric in metrics],
            }
        )

        print(metrics_df)

        # np.log(np.cumprod(1 + strategy_return)).plot()
        # plt.ylabel("Log Price")
        np.cumprod(1 + strategy_return).plot()
        plt.ylabel("Price / Price(0)")
        plt.xlabel("Date")
        plt.show()


def main():

    bs = backtest()
    alpha_df = bs.csv_to_df(alpha_path)
    alpha_df.drop(exclude, axis=1, inplace=True)
    bs.run(alpha_df)


if __name__ == "__main__":
    main()
