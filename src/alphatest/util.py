import numpy as np
import pandas as pd


class cross_sectional:

    def __init__(self, valid_df: pd.DataFrame):
        self.valid_df = valid_df

    @staticmethod
    def neutralize(row):
        """
        Shifts a row's finite values so their mean is 0 (neutralizing them).

        This function ignores NaN and infinity values during the calculation
        and leaves them unchanged in the output.
        """

        # Calculate the mean of *only* the finite values by temporarily
        # replacing infinity and -infinity with NaN.
        # mean_of_finite_values = row.mask(np.isinf(row), np.nan).mean()
        mean_of_finite_values = (
            row.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).mean()
        )

        # Subtracting the mean from the original row centers the finite numbers
        # while leaving inf, -inf, and NaN values untouched.
        return row - mean_of_finite_values

    @staticmethod
    def scale_final(row):
        """
        Scales a row so that the sum of its absolute values is 1.

        Infinity (inf) and NaN values are converted to 0 before scaling.
        """
        # Step 1: Replace inf and -inf with NaN, then replace all NaN with 0
        # row_no_nan = row.mask(np.isinf(row), np.nan).fillna(0)
        row_no_nan = (
            row.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).fillna(0)
        )

        # Step 2: Calculate the sum of the absolute values (L1 norm)
        l1_norm = row_no_nan.abs().sum()

        # Step 3: Handle the edge case where the sum is 0 to avoid division by zero
        if l1_norm == 0:
            return row_no_nan

        # Step 4: Divide each element by the L1 norm
        return row_no_nan / l1_norm

    @staticmethod
    def rank(row):
        """
        Calculates the percentile rank of each value in a pandas Series (a row).

        - Ranks are scaled between 0 and 1.
        - None/NaN values are preserved and ignored during ranking.
        - In case of ties, the average rank is assigned.

        Args:
            row (pd.Series): A row from a pandas DataFrame.

        Returns:
            pd.Series: A new Series with the percentile ranks.
        """
        # The rank method with pct=True computes ranks as percentiles (0 to 1)
        # na_option='keep' ensures that NaN values in the input remain NaN in the output
        return row.rank(method="average", pct=True, na_option="keep")
