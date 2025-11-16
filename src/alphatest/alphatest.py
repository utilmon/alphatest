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
        crypto_benchmarks: list = ["BTC"],
        stock_benchmarks: list = ["VOO", "GLD"],
        compound: bool = True,
    ):
        """
        Initializes the class with configuration parameters for data analysis and benchmarking.

        Args:
            data_path (str): Path to the data source.
            annual_days (int, optional): Number of trading days in a year (default is 365).
            daily_ticks (int, optional): Number of ticks per day (default is 1).
            risk_free_return (float, optional): Annualized risk-free return rate (default is 0.03).
            trading_fee (float, optional): Trading fee per transaction (default is 0.00025).
            crypto_benchmarks (list, optional): List of cryptocurrency benchmark symbols (default is ["BTC"]).
            stock_benchmarks (list, optional): List of stock benchmark symbols (default is ["VOO", "GLD"]).
            compound (bool, optional): Whether to use compound returns (default is True).

        Attributes:
            annual_ticks (int): Total number of ticks in a year.
            daily_ticks (int): Number of ticks per day.
            fee (float): Trading fee per transaction.
            risk_free_return (float): Annualized risk-free return rate.
            data_path (str): Path to the data source.
            valid_df (DataFrame): Validated data frame generated from the data source.
            cs (object): Custom object initialized with the validated data frame.
            crypto_benchmarks (list): Cryptocurrency benchmark symbols.
            stock_benchmarks (list): Stock benchmark symbols.
            compound (bool): Whether to use compound returns.

        Calls:
            self.generate_valid_df(): Generates a validated data frame from the data source.
            self.print_parameters(): Prints the initialized parameters.
        """
        self.annual_ticks = annual_days * daily_ticks
        self.daily_ticks = daily_ticks
        self.fee = trading_fee
        self.risk_free_return = risk_free_return
        self.data_path = data_path
        self.valid_df = self.generate_valid_df()
        self.cs = cs(valid_df=self.valid_df)
        self.crypto_benchmarks = crypto_benchmarks
        self.stock_benchmarks = stock_benchmarks
        self.compound = compound
        self.print_parameters()

    def print_parameters(self):
        """Print a human-readable summary of the backtesting parameters.

        This method writes a formatted summary of the instance's backtesting configuration
        to standard output. The printed fields include:

        - annual_ticks (int): number of trading ticks per year.
        - daily_ticks (int): number of ticks per trading day.
        - fee (float): trading fee as a fractional value (printed as a percentage).
        - risk_free_return (float): risk-free return as a fractional value (printed as a percentage).
        - data_path (str): filesystem path to input data.
        - crypto_benchmarks (iterable): list or iterable of cryptocurrency benchmark identifiers.
        - stock_benchmarks (iterable): list or iterable of stock benchmark identifiers.

        Side effects:
        - Outputs formatted lines to stdout via print().

        Returns:
        - None
        """
        print("Backtesting parameters:")
        print(
            f"Annual days: {self.annual_ticks/self.daily_ticks}, Daily ticks: {self.daily_ticks}"
        )
        print(f"Trading fee (%): {self.fee * 100}%")
        print(f"Risk free return (%): {self.risk_free_return *100}%")
        print(rf"Data path: {self.data_path}")
        print(f"Crypto benchmarks: {self.crypto_benchmarks}")
        print(f"Stock benchmarks: {self.stock_benchmarks}")

    def generate_valid_df(self):
        """
        Generate a mask indicating which return observations are valid (non-missing).
        This method retrieves the "returns" data via self.get_data("returns") and
        produces a structure of the same shape where each element is 1 if the
        corresponding return value is not missing (as determined by pandas.notna),
        and numpy.nan if the return value is missing.
        Returns
        -------
        pandas.Series or pandas.DataFrame
            A structure matching the shape and index (and columns, if applicable) of
            the original returns data, containing 1 for valid (non-NA) entries and
            numpy.nan for missing entries.
        Notes
        -----
        - The implementation assumes the returned `returns` object supports elementwise
          mapping (e.g., a pandas.Series). If a DataFrame is supplied, elementwise
          mapping may require a different API (such as DataFrame.applymap).
        - Uses pandas.notna to detect non-missing values and numpy.nan to represent missing output.
        Example
        -------
        # If self.get_data("returns") yields a Series: [0.02, NaN, -0.01]
        # The returned mask will be: [1.0, NaN, 1.0]
        """

        ### for now, will use returns df
        returns_df = self.get_data("returns")
        df = returns_df.map(lambda x: 1 if pd.notna(x) else np.nan)
        return df

    def get_data(self, data_type: str, shift: int = 0):
        """
        Load a CSV file for the given data type, parse its "time" column as datetimes,
        set that column as the DataFrame index, and return the DataFrame shifted by
        the given number of periods.

        Parameters
        ----------
        data_type : str
            Base name of the CSV file (without extension) located under self.data_path.
            The method will open "<self.data_path>/<data_type>.csv".
        shift : int, optional
            Number of periods to shift the data (passed to pandas.DataFrame.shift).
            Positive values shift data forward (introducing NaNs), negative values
            shift data backward. Default is 0 (no shift).

        Returns
        -------
        pandas.DataFrame
            The loaded DataFrame indexed by a DatetimeIndex derived from the "time"
            column, with values shifted by `shift` periods.

        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist at the constructed path.
        pandas.errors.EmptyDataError, pandas.errors.ParserError, ValueError
            If the CSV cannot be parsed or the "time" column is missing/invalid.

        Notes
        -----
        - The "time" column is converted with pandas.to_datetime; timezone/format
          inference follows pandas rules.
        - pandas.DataFrame.shift shifts the data values, not the index.
        - Caller is responsible for ensuring self.data_path is defined and accessible.

        Example
        -------
        >>> df = self.get_data("sensor_readings", shift=1)
        """
        data_path = os.path.join(self.data_path, data_type + ".csv")
        df = pd.read_csv(data_path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        return df.shift(shift)

    def compute_turnover(self, alpha_df: pd.DataFrame):
        """
        Computes the turnover for each row in the given alpha DataFrame.

        Turnover is calculated as the sum of the absolute differences of alpha values
        across columns for each row compared to the previous row.

        Args:
            alpha_df (pd.DataFrame): A DataFrame containing alpha values, where each row
                represents a time period and each column represents an asset.

        Returns:
            pd.Series: A Series containing the turnover value for each row (time period).
        """
        diff_df = alpha_df.diff()
        turnover = diff_df.abs().sum(axis=1)
        return turnover

    def compute_position(self, alpha_df: pd.DataFrame):
        """
        Computes the mean of the row-wise sums of the given DataFrame.

        Args:
            alpha_df (pd.DataFrame): Input DataFrame containing alpha values.

        Returns:
            float: The mean of the sums computed across each row.
        """
        return alpha_df.sum(axis=1).mean()

    def compute_compounding_metrics(
        self, strategy_return: pd.Series, annual_ticks: int
    ):
        """
        Compute a set of performance and risk metrics from a series of periodic strategy returns.
        Parameters
        ----------
        strategy_return : pandas.Series
            Series of periodic simple returns (e.g. 0.01 for +1%). The index is preserved
            for the returned drawdown series. NaNs are not explicitly dropped by the implementation,
            so callers may want to clean the series beforehand.
        annual_ticks : int
            Number of return periods per year (e.g. 252 for daily, 12 for monthly, etc.).
        Returns
        -------
        dict
            A dictionary with the following keys and value types:
            - "annual_return" (float):
                Annualized geometric return computed as (prod(1 + r))^(annual_ticks / total_ticks) - 1,
                where total_ticks is the number of observations in strategy_return.
            - "annual_vol" (float):
                Annualized standard deviation computed as std(strategy_return) * sqrt(annual_ticks).
            - "sharpe" (float or np.nan):
                Annualized Sharpe ratio computed as (annual_return - self.risk_free_return) / annual_vol.
                If annual_vol == 0 the value is np.nan. Note: this uses the instance attribute
                self.risk_free_return as the risk-free rate (assumed annualized).
            - "profit_factor" (float):
                Sum of positive returns divided by the absolute sum of negative returns:
                sum(returns[returns > 0]) / -sum(returns[returns < 0]).
                If there are no negative returns the denominator is zero and the current implementation
                will raise a division-by-zero error or produce an infinite value.
            - "sortino" (float or np.nan):
                Sortino ratio computed as (annual_return - self.risk_free_return) / downside_std,
                where downside_std = std(returns[returns < 0]) * sqrt(annual_ticks).
                If downside_std == 0 the value is np.nan.
            - "drawdown" (pandas.Series):
                Cumulative returns series computed as (1 + returns).cumprod(), then drawdown series
                computed as cum_returns / cum_returns.cummax() - 1. Has the same index and length
                as strategy_return.
            - "max_drawdown" (float):
                Minimum (most negative) value of the drawdown series.
            - "calmar" (float or np.nan):
                Calmar ratio computed as (annual_return - self.risk_free_return) / -max_drawdown.
                If max_drawdown == 0 the value is np.nan.
        Notes
        -----
        - The function assumes input returns are simple returns (not log returns) and uses geometric compounding.
        - Edge cases:
          - Empty strategy_return or total_ticks == 0 will lead to errors in the current implementation.
          - NaNs in strategy_return may propagate into computed metrics.
          - Profit factor denominator or volatility denominators equal to zero are handled for Sharpe/Sortino/Calmar
            by returning np.nan, but profit_factor is not guarded against division by zero.
        """

        metrics_map = {}
        total_ticks = strategy_return.shape[0]
        cumulative = (strategy_return + 1).prod()
        annual_return = (cumulative ** (annual_ticks / total_ticks)) - 1
        # annual_return = strategy_return.mean() * annual_ticks
        metrics_map["annual_return"] = annual_return
        annual_vol = np.sqrt(annual_ticks) * strategy_return.std()
        metrics_map["annual_vol"] = annual_vol
        metrics_map["sharpe"] = (
            (annual_return - self.risk_free_return) / annual_vol
            if annual_vol != 0
            else np.nan
        )
        metrics_map["profit_factor"] = (
            strategy_return[strategy_return > 0].sum()
            / -strategy_return[strategy_return < 0].sum()
        )
        downside_std = strategy_return[strategy_return < 0].std() * np.sqrt(
            annual_ticks
        )
        metrics_map["sortino"] = (
            (annual_return - self.risk_free_return) / downside_std
            if downside_std != 0
            else np.nan
        )
        cum_returns = (strategy_return + 1).cumprod()
        drawdown = cum_returns / cum_returns.cummax() - 1
        metrics_map["drawdown"] = drawdown
        max_drawdown = drawdown.min()
        metrics_map["max_drawdown"] = max_drawdown
        metrics_map["calmar"] = (
            (annual_return - self.risk_free_return) / -max_drawdown
            if max_drawdown != 0
            else np.nan
        )
        return metrics_map

    def compute_additive_metrics(self, strategy_return: pd.Series, annual_ticks: int):
        """
        Compute a set of performance and risk metrics for a strategy return series.
        Parameters
        ----------
        strategy_return : pandas.Series
            A time-ordered series of periodic strategy returns (can be arithmetic returns).
            The series index (timestamps) is preserved for metrics that are time series (e.g., drawdown).
        annual_ticks : int
            Number of return observations per year (used to annualize mean and volatility).
            For example, use 252 for daily returns, 12 for monthly returns, etc.
        Returns
        -------
        dict
            A dictionary mapping metric names to their computed values:
            - "annual_return" (float): Annualized mean return = mean(strategy_return) * annual_ticks.
            - "annual_vol" (float): Annualized volatility = std(strategy_return) * sqrt(annual_ticks).
            - "sharpe" (float or numpy.nan): Sharpe ratio computed as (annual_return - self.risk_free_return) / annual_vol;
              returns np.nan when annual_vol is zero.
            - "profit_factor" (float): Ratio of gross profits to gross losses:
                sum(positive returns) / -sum(negative returns).
              Note: if there are no losses (denominator is zero) this value may be infinite or undefined depending on numeric types.
            - "sortino" (float or numpy.nan): Sortino ratio computed as (annual_return - self.risk_free_return) / downside_std,
              where downside_std is the annualized standard deviation of negative returns (std(negatives) * sqrt(annual_ticks));
              returns np.nan when downside_std is zero.
            - "drawdown" (pandas.Series): Time series of rolling drawdown values computed from cumulative returns:
              cum_returns = strategy_return.cumsum() + 1, drawdown = cum_returns / cum_returns.cummax() - 1.
              This series has the same index as strategy_return.
            - "max_drawdown" (float): The maximum (most negative) drawdown observed = min(drawdown).
            - "calmar" (float or numpy.nan): Calmar ratio computed as (annual_return - self.risk_free_return) / -max_drawdown;
              returns np.nan when max_drawdown is zero.
        Notes
        -----
        - The method references self.risk_free_return when computing Sharpe, Sortino and Calmar ratios; ensure that attribute is set.
        - Annualization assumes independent, identically distributed returns at the frequency implied by annual_ticks.
        - Returns are computed using simple arithmetic statistics; if using log returns or other conventions, adapt inputs accordingly.
        """

        metrics_map = {}
        annual_return = strategy_return.mean() * annual_ticks
        metrics_map["annual_return"] = annual_return
        annual_vol = np.sqrt(annual_ticks) * strategy_return.std()
        metrics_map["annual_vol"] = annual_vol

        metrics_map["sharpe"] = (
            (annual_return - self.risk_free_return) / annual_vol
            if annual_vol != 0
            else np.nan
        )
        metrics_map["profit_factor"] = (
            strategy_return[strategy_return > 0].sum()
            / -strategy_return[strategy_return < 0].sum()
        )
        downside_std = strategy_return[strategy_return < 0].std() * np.sqrt(
            annual_ticks
        )
        metrics_map["sortino"] = (
            (annual_return - self.risk_free_return) / downside_std
            if downside_std != 0
            else np.nan
        )
        cum_returns = strategy_return.cumsum() + 1
        drawdown = cum_returns / cum_returns.cummax() - 1
        metrics_map["drawdown"] = drawdown
        max_drawdown = drawdown.min()
        metrics_map["max_drawdown"] = max_drawdown
        metrics_map["calmar"] = (
            (annual_return - self.risk_free_return) / -max_drawdown
            if max_drawdown != 0
            else np.nan
        )
        return metrics_map

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
        """
        Plot a histogram of numeric values contained in a pandas DataFrame.
        This method flattens the DataFrame's underlying values, removes NaN entries,
        uses self.get_optimal_bins(...) to choose histogram bins, and displays a
        Matplotlib histogram with labeled axes and a title that reports the mean (μ)
        and standard deviation (σ) of the plotted values.
        Parameters
        ----------
        alpha_df : pandas.DataFrame
            Input DataFrame whose values will be flattened and treated as numeric data.
            Non-numeric entries that cannot be converted to numeric types will typically
            become NaN and be ignored. An empty DataFrame or one containing only NaNs
            will produce no plot.
        Returns
        -------
        None
            This method has the side effect of showing a Matplotlib figure. If no
            numerical data is found, it prints a message ("No numerical data was found
            to plot the distribution") and returns without displaying a plot.
        Raises
        ------
        TypeError
            If alpha_df is not a pandas.DataFrame, operations that assume a DataFrame
            may raise a TypeError or AttributeError.
        Notes
        -----
        - The number of histogram bins is determined by calling self.get_optimal_bins(final_values).
        - Mean and standard deviation shown in the title are computed with numpy and reflect
          the filtered values after NaN removal.
        - This method relies on matplotlib.pyplot for plotting and numpy for numerical operations.
        """

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
        """
        Execute the alpha strategy backtest and generate performance metrics and visualizations.
        Parameters
        ----------
        alpha_df : pd.DataFrame
            DataFrame containing alpha signals/positions for each asset over time.
            Can also accept a pd.Series, which will be converted to a DataFrame.
        scale_final : bool, optional
            Whether to apply final scaling to the alpha values. Default is True.
            If alpha_df is a Series, this is automatically set to False.
        Returns
        -------
        dict
            Dictionary containing strategy performance metrics with keys:
            - 'sharpe': Sharpe ratio
            - 'profit_factor': Profit factor
            - 'annual_return': Annual return (CAGR)
            - 'max_drawdown': Maximum drawdown
            - 'annual_vol': Annualized volatility
            - 'sortino': Sortino ratio
            - 'calmar': Calmar ratio
            - 'returns': pd.Series of strategy returns
            - 'turnover': Mean daily turnover (%)
            - 'long_short': Long/short position ratio
            - 'drawdown': Series of drawdown values
        Notes
        -----
        - Prints a metrics comparison table including benchmarks (stock and crypto)
        - Generates a 2-panel matplotlib figure:
            - Top panel: Log-scale cumulative returns comparison
            - Bottom panel: Drawdown comparison
        - Handles NaN values and infinite values appropriately
        - Accounts for transaction fees based on turnover
        - Aligns alpha signals with returns data using common indices
        """

        self.print_parameters()
        print(f"Scale final: {scale_final}\n")

        if type(alpha_df) == pd.Series:
            alpha_df = alpha_df.to_frame()
            scale_final = False

        ### remove NaN only rows
        first_valid_index = alpha_df.first_valid_index()
        if first_valid_index is not None:
            alpha_df = alpha_df.loc[first_valid_index:]

        ### neutralize

        alpha_df = alpha_df.apply(pd.to_numeric, errors="coerce")

        if scale_final:
            alpha_df = alpha_df.apply(self.cs.scale_final, axis=1)
        else:
            alpha_df = alpha_df.mask(np.isinf(alpha_df), np.nan).fillna(0)
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
        if self.compound:
            metrics_map = self.compute_compounding_metrics(
                strategy_return, self.annual_ticks
            )
        else:
            metrics_map = self.compute_additive_metrics(
                strategy_return, self.annual_ticks
            )
        metrics_map["returns"] = strategy_return
        metrics_map["turnover"] = turnover.mean() * self.daily_ticks * 100
        metrics_map["long_short"] = self.compute_position(alpha_df)

        parameters_to_show = [
            "sharpe",
            "profit_factor",
            "annual_return",
            "max_drawdown",
            "annual_vol",
            "sortino",
            "calmar",
            "turnover",
            "long_short",
        ]

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
                "Strategy": [
                    f"{metrics_map[param]:.2f}" for param in parameters_to_show
                ],
            }
        )

        price_map = {}
        drawdown_map = {}
        strategy_index = strategy_return.index
        start, end = strategy_index[0], strategy_index[-1]
        # strategy_return.plot()

        ### benchmarks
        benchmarks_df = self.get_data("../benchmarks/returns")
        for benchmark in self.stock_benchmarks + self.crypto_benchmarks:
            df = benchmarks_df[benchmark].dropna()
            df = df[(df.index >= start) & (df.index <= end)]
            # df = df[start:end]
            if benchmark in self.stock_benchmarks:
                annual_ticks = 252
            elif benchmark in self.crypto_benchmarks:
                annual_ticks = 365
            bench_metrics = self.compute_compounding_metrics(
                df, annual_ticks=annual_ticks
            )
            bench_metrics["turnover"] = 0
            bench_metrics["long_short"] = 1

            metrics_df[benchmark] = [
                f"{bench_metrics[param]:.2f}" for param in parameters_to_show
            ]
            price_map[benchmark] = np.cumprod(1 + df)
            drawdown_map[benchmark] = bench_metrics["drawdown"]

        print(metrics_df)

        fig, axs = plt.subplots(
            nrows=2, ncols=1, gridspec_kw={"height_ratios": [2, 1]}, figsize=(11, 12)
        )

        if self.compound:
            price_map["strategy"] = np.cumprod(1 + strategy_return)
        else:
            price_map["strategy"] = 1 + strategy_return.cumsum()
        df_price = pd.DataFrame(price_map).sort_index()
        # print(df_price)

        df_price.ffill().plot(ax=axs[0])
        axs[0].set_ylabel("Growth of $1 investment")
        # axs[0].set_xlabel("Date")
        if self.compound:
            axs[0].set_yscale("log")

        drawdown_map["strategy"] = metrics_map["drawdown"]
        df_drawdown = pd.DataFrame(drawdown_map).sort_index()

        df_drawdown.ffill().plot(ax=axs[1])
        axs[1].set_ylabel("Drawdown")
        axs[1].set_xlabel("Date")

        # plt.show()

        return metrics_map
