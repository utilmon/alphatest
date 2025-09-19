import pandas as pd
from .util import cross_sectional as cs
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["figure.figsize"] = (12, 6)


class backtest:
    def __init__(
        self,
        data_path: str,
        annual_days: int = 365,
        risk_free_return: float = 0.03,
        trading_fee: float = 0.00025,
    ):
        self.annual_days = annual_days
        self.fee = trading_fee
        self.risk_free_return = risk_free_return
        self.data_path = data_path
        self.valid_df = self.generate_valid_df()
        self.cs = cs(valid_df=self.valid_df)

    def generate_valid_df(self):

        ### for now, will use returns df
        returns_df = self.get_data("returns")
        df = returns_df.map(lambda x: 1 if pd.notna(x) else np.nan)
        return df

    def csv_to_df(self, path: str):
        return pd.read_csv(path, index_col="date")

    def get_data(self, data_type: str, shift: int = 0):
        data_path = os.path.join(self.data_path, data_type + ".csv")
        df = pd.read_csv(data_path).shift(shift)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        return df

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
        return [sharpe, annual_return, max_drawdown, annual_vol, sortino, calmar]

    def run(self, alpha_df: pd.DataFrame):

        alpha_df = alpha_df.apply(self.cs.scale_final, axis=1)
        returns_df = self.get_data("returns")
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
                    "Sharpe Ratio",
                    "CAGR",
                    "Maximum Drawdown",
                    "Annualized Volatility",
                    "Sortino Ratio",
                    "Calmar Ratio",
                    "Annual Turnover (%)",
                    "Long/Short position",
                ],
                "Strategy": [f"{metric:.2f}" for metric in metrics],
            }
        )

        price_map = {}
        strategy_index = strategy_return.index
        start, end = strategy_index[0], strategy_index[-1]
        # strategy_return.plot()

        ### benchmarks
        benchmarks_df = self.get_data("../benchmarks/returns")
        for benchmark in benchmarks_df.columns:
            df = benchmarks_df[benchmark].dropna()
            # df = df[(df.index >= start) & (df.index <= end)]
            df = df[start:end]
            metrics = self.compute_metrics(df)
            metrics.extend([0, 1])
            metrics_df[benchmark] = [f"{metric:.2f}" for metric in metrics]
            price_map[benchmark] = np.cumprod(1 + df).rename(benchmark)

        print(metrics_df)

        price_map["strategy"] = np.cumprod(1 + strategy_return).rename("strategy")
        df_price = pd.concat(price_map.values(), axis=1)
        df_price = df_price.sort_index()
        # print(df_price)

        ax = df_price.ffill().plot()
        # date_format = mdates.AutoDateFormatter(mdates.MonthLocator())
        # ax.xaxis.set_major_formatter(date_format)
        plt.ylabel("Price / Price(0)")
        plt.xlabel("Date")
        plt.yscale("log")
        plt.legend()
        plt.show()
