import pandas as pd
from .util import cross_sectional as cs
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["figure.figsize"] = (11, 6)


class backtest:
    def __init__(
        self,
        data_path: str,
        annual_days: int = 365,  # 365, 252
        daily_ticks: int = 1,  # 1 or 24
        risk_free_return: float = 0.03,
        trading_fee: float = 0.00025,
    ):
        self.annual_ticks = annual_days * daily_ticks
        self.daily_ticks = daily_ticks
        self.fee = trading_fee
        self.risk_free_return = risk_free_return
        self.data_path = data_path
        self.valid_df = self.generate_valid_df()
        self.cs = cs(valid_df=self.valid_df)
        self.crypto_benchmarks = ["BTC"]
        self.stock_benchmarks = ["VOO", "GLD"]

    def generate_valid_df(self):

        ### for now, will use returns df
        returns_df = self.get_data("returns")
        df = returns_df.map(lambda x: 1 if pd.notna(x) else np.nan)
        return df

    def get_data(self, data_type: str, shift: int = 0):
        data_path = os.path.join(self.data_path, data_type + ".csv")
        df = pd.read_csv(data_path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        return df.shift(shift)

    def compute_turnover(self, alpha_df: pd.DataFrame):
        diff_df = alpha_df.diff()
        turnover = diff_df.abs().sum(axis=1)
        return turnover

    def compute_position(self, alpha_df: pd.DataFrame):
        return alpha_df.sum(axis=1).mean()

    def compute_metrics(self, strategy_return: pd.Series, annual_ticks: int):
        total_ticks = strategy_return.shape[0]
        cumulative = (strategy_return + 1).prod()
        annual_return = (cumulative ** (annual_ticks / total_ticks)) - 1
        # annual_return = strategy_return.mean() * annual_ticks
        annual_vol = np.sqrt(annual_ticks) * strategy_return.std()
        sharpe = (
            (annual_return - self.risk_free_return) / annual_vol
            if annual_vol != 0
            else np.nan
        )
        profit_factor = (
            strategy_return[strategy_return > 0].sum()
            / -strategy_return[strategy_return < 0].sum()
        )
        downside_std = strategy_return[strategy_return < 0].std() * np.sqrt(
            annual_ticks
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
        return [
            sharpe,
            profit_factor,
            annual_return,
            max_drawdown,
            annual_vol,
            sortino,
            calmar,
        ]

    def get_optimal_bins(self, array: np.array):
        """
        Freedman-Diaconis Rule
        """

        if array.size < 2:
            return 1
        q75, q25 = np.percentile(array, [75, 25])
        iqr = q75 - q25

        # calculate bin width
        bin_width = 2 * iqr / (len(array) ** (1 / 3))
        if bin_width == 0:
            return int(np.sqrt(len(array)))

        array_range = array.max() - array.min()
        return int(np.ceil(array_range / bin_width))

    def get_distribution(self, alpha_df: pd.DataFrame):

        numerical_values = alpha_df.values.flatten()
        if numerical_values.size == 0:
            print("No numerical data was found to plot the distribution")
            return

        final_values = numerical_values[~np.isnan(numerical_values)]
        plt.hist(
            final_values,
            bins=self.get_optimal_bins(final_values),
        )
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title(
            rf"$\mu$:{np.mean(final_values):.3g}, $\sigma$:{np.std(final_values):.3g}"
        )
        plt.show()

    def run(self, alpha_df: pd.DataFrame, scale_final: bool = True):

        if type(alpha_df) == pd.Series:
            alpha_df = alpha_df.to_frame()
            scale_final = False

        ### remove NaN only rows
        first_valid_index = alpha_df.first_valid_index()
        if first_valid_index is not None:
            alpha_df = alpha_df.loc[first_valid_index:]

        ### neutralize

        if scale_final:
            alpha_df = alpha_df.apply(self.cs.scale_final, axis=1)
        else:
            alpha_df = alpha_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        returns_df = self.get_data("returns")
        common_rows = alpha_df.shift(1).index.intersection(returns_df.index)
        strategy_return = (
            (returns_df.loc[common_rows, alpha_df.columns] * alpha_df.shift(1))
            .fillna(0)
            .sum(axis=1)
            .rename("strategy")
        )
        turnover = self.compute_turnover(alpha_df)
        strategy_return -= turnover * self.fee
        metrics = self.compute_metrics(strategy_return, self.annual_ticks)
        daily_turnover = turnover.mean() * self.daily_ticks
        long_short = self.compute_position(alpha_df)
        metrics.extend([daily_turnover * 100, long_short])

        metrics_df = pd.DataFrame(
            {
                "Metrics": [
                    "Sharpe Ratio",
                    "Profit Factor",
                    "CAGR",
                    "Maximum Drawdown",
                    "Annualized Volatility",
                    "Sortino Ratio",
                    "Calmar Ratio",
                    "Daily Turnover (%)",
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
        for benchmark in self.crypto_benchmarks + self.stock_benchmarks:
            df = benchmarks_df[benchmark].dropna()
            # df = df[(df.index >= start) & (df.index <= end)]
            df = df[start:end]
            if benchmark in self.stock_benchmarks:
                annual_ticks = 252
            elif benchmark in self.crypto_benchmarks:
                annual_ticks = 365
            metrics = self.compute_metrics(df, annual_ticks=annual_ticks)

            metrics.extend([0, 1])
            metrics_df[benchmark] = [f"{metric:.2f}" for metric in metrics]
            price_map[benchmark] = np.cumprod(1 + df)

        print(metrics_df)

        price_map["strategy"] = np.cumprod(1 + strategy_return)
        df_price = pd.DataFrame(price_map).sort_index()
        # print(df_price)

        ax = df_price.ffill().plot()
        # date_format = mdates.AutoDateFormatter(mdates.MonthLocator())
        # ax.xaxis.set_major_formatter(date_format)
        plt.ylabel("Growth of $1 investment")
        plt.xlabel("Date")
        plt.yscale("log")
        plt.legend()
        plt.show()

        return strategy_return
