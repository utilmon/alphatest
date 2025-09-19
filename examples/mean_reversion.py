#%%
from alphatest import backtest, util

data_path = r"C:\Users\qwane\Documents\git\alphatest\data\binance_spot"

at = backtest(data_path=data_path)
df_returns = at.get_data("returns")

#%%
at.run(df_returns)