# Import the libraries from the requirements.txt file 
import yfinance as yf
from pytrends.request import TrendReq
import pandas as pd
import numpy as np

# We need to grab the OHLC data for btc-usd for the last 5 years, today is 2026-02-28, so we will grab data from 2021-02-28 to 2026-02-28, and we need to make sure it is on a daily frequency.
btc_data = yf.download('BTC-USD', start='2021-02-28', end='2026-02-28', interval='1d', auto_adjust=True)

# We also need to grab the google trends data for the keyword "bitcoin" for the same time period, and we need to make sure it is on a daily frequency as well.
pytrends = TrendReq(hl='en-US', tz=360)
pytrends.build_payload(kw_list=['bitcoin'], timeframe='2021-02-28 2026-02-28', geo='', gprop='')
trends_data = pytrends.interest_over_time()

# We need to convert the daily Bitcon price data into a time series of daily logarithmic returns. 
btc_data['Log_Returns'] = np.log(btc_data['Close'] / btc_data['Close'].shift(1))

# We need to make the google trends data stationary, and to do this we have to calculate the change from the day before, so todays value minus yesterdays value, and we will call this "Trends_Change", this should transform the data into a series of search interest levels into a series of daily changes
trends_data['Trends_Change'] = trends_data['bitcoin'].diff()

# Now we need to use transfer entropy calculation which is going to use PyInform library, and we will also need to test both directions, meaning we have to calculate the information flow from Google Trends to Log Returns, and we are also going to calculate the information flow from Log Returns to Google Trends, and we will call these "TE_Trends_to_Returns" and "TE_Returns_to_Trends" respectively.
from pyinform import transfer_entropy

# Align both series on the same dates so source/target have identical shape
aligned_data = pd.concat(
    [
        btc_data['Log_Returns'].rename('Log_Returns'),
        trends_data['Trends_Change'].rename('Trends_Change')
    ],
    axis=1,
    join='inner'
).dropna()


def discretize_series(series, bins=5):
    unique_count = series.nunique()
    if unique_count < 2:
        raise ValueError("Not enough variation to discretize series for transfer entropy.")
    effective_bins = min(bins, unique_count)
    return pd.qcut(
        series.rank(method='first'),
        q=effective_bins,
        labels=False
    ).astype(int)


returns_states = discretize_series(aligned_data['Log_Returns'])
trends_states = discretize_series(aligned_data['Trends_Change'])

# Calculate the transfer entropy from Google Trends to Log Returns
TE_Trends_to_Returns = transfer_entropy(trends_states.to_numpy(), returns_states.to_numpy(), k=1)
# Calculate the transfer entropy from Log Returns to Google Trends
TE_Returns_to_Trends = transfer_entropy(returns_states.to_numpy(), trends_states.to_numpy(), k=1)

# Now we need to repeat the transfer entropy calculations for different time delays (lags), from 1 day up to at least 6 days, to see how long any predictive information lasts. 
for lag in range(1, 7):
    TE_Trends_to_Returns_lag = transfer_entropy(trends_states.to_numpy(), returns_states.to_numpy(), k=lag)
    TE_Returns_to_Trends_lag = transfer_entropy(returns_states.to_numpy(), trends_states.to_numpy(), k=lag)
    print(f"Lag {lag} days: TE_Trends_to_Returns = {TE_Trends_to_Returns_lag}, TE_Returns_to_Trends = {TE_Returns_to_Trends_lag}")

# Now we need to be able to interpret the transfer entropy results and determine the relationship of GoogleTrends Bitcoin index is a leading indicator of market activity, or vice-versa
if TE_Trends_to_Returns > TE_Returns_to_Trends:
    print("Directional result: Trends -> Returns shows stronger information flow.")
elif TE_Returns_to_Trends > TE_Trends_to_Returns:
    print("Directional result: Returns -> Trends shows stronger information flow.")
else:
    print("Directional result: Both directions show equal information flow.")
