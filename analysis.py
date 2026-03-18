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


def permutation_te_pvalue(source, target, observed_te, k=1, n_shuffles=500, seed=None):
    """Estimate a p-value for an observed transfer entropy via permutation testing.

    The source series is randomly shuffled *n_shuffles* times, and the transfer
    entropy is recomputed for each shuffle.  The p-value is the fraction of
    shuffled TE values that are greater than or equal to the observed value.
    A small p-value (< 0.05) indicates that the observed TE is unlikely under
    the null hypothesis of no information flow from source to target.

    Pass an integer *seed* for reproducible results; leave as None for a
    different random draw on each call.
    """
    rng = np.random.default_rng(seed=seed)
    source_array = source.to_numpy().copy()
    target_array = target.to_numpy()
    count_geq = 0
    for _ in range(n_shuffles):
        shuffled = rng.permutation(source_array)
        shuffled_te = transfer_entropy(shuffled, target_array, k=k)
        if shuffled_te >= observed_te:
            count_geq += 1
    return count_geq / n_shuffles


# Compute p-values via permutation testing for the base (k=1) TE estimates
print("\nComputing significance via permutation testing (500 shuffles each)…")
pvalue_trends_to_returns = permutation_te_pvalue(trends_states, returns_states, TE_Trends_to_Returns, k=1)
pvalue_returns_to_trends = permutation_te_pvalue(returns_states, trends_states, TE_Returns_to_Trends, k=1)

print(f"\nTE Trends -> Returns: {TE_Trends_to_Returns:.6f}  (p-value: {pvalue_trends_to_returns:.4f})")
print(f"TE Returns -> Trends: {TE_Returns_to_Trends:.6f}  (p-value: {pvalue_returns_to_trends:.4f})")

# Interpret the results using the framework described in the README
SIGNIFICANCE_THRESHOLD = 0.05
sig_trends_to_returns = pvalue_trends_to_returns < SIGNIFICANCE_THRESHOLD
sig_returns_to_trends = pvalue_returns_to_trends < SIGNIFICANCE_THRESHOLD

print()
if sig_trends_to_returns and not sig_returns_to_trends:
    # Case A: The "Exploitable Pattern" (Unidirectional: Trends -> Returns)
    print("Case A: The 'Exploitable Pattern' (Unidirectional: Trends -> Returns)")
    print("A meaningful, one-way information flow from search interest to price returns exists.")
    print("This suggests search interest has predictive value for future returns.")
elif sig_returns_to_trends and not sig_trends_to_returns:
    # Case B: Market Activity Drives Attention (Unidirectional: Returns -> Trends)
    print("Case B: Market Activity Drives Attention (Unidirectional: Returns -> Trends)")
    print("Price changes are driving public attention, but public attention is not providing a predictive signal for price.")
elif sig_trends_to_returns and sig_returns_to_trends:
    # Case C: A Two-Way Relationship (Bidirectional)
    print("Case C: A Two-Way Relationship (Bidirectional)")
    if TE_Trends_to_Returns > TE_Returns_to_Trends:
        print("The dominant direction is Trends -> Returns (higher TE score).")
        print("Directional result: Trends -> Returns shows stronger information flow.")
    elif TE_Returns_to_Trends > TE_Trends_to_Returns:
        print("The dominant direction is Returns -> Trends (higher TE score).")
        print("Directional result: Returns -> Trends shows stronger information flow.")
    else:
        print("Both directions show equal information flow.")
else:
    # Case D: No Pattern Found
    print("Case D: No Pattern Found")
    print("There is no evidence of a statistically significant predictive relationship in either direction.")
