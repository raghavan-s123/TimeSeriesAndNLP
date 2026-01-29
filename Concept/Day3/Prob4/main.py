import warnings
import os
import sys
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

file = input().strip()
df = pd.read_csv(os.path.join(sys.path[0], file))

df['Datetime'] = pd.to_datetime(df['Datetime'])

df = df.dropna(subset=['Power_Consumption_diff'])

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

print(f"Training data size: {len(train)}")
print(f"Testing data size: {len(test)}\n")

series = train['Power_Consumption_diff']

ar_aic = {}
ma_aic = {}

for p in range(1, 6):
    model = ARIMA(series, order=(p, 0, 0))
    result = model.fit()
    ar_aic[p] = result.aic
    print(f"AR({p}) AIC: {result.aic}")

for q in range(1, 6):
    model = ARIMA(series, order=(0, 0, q))
    result = model.fit()
    ma_aic[q] = result.aic
    print(f"MA({q}) AIC: {result.aic}")

best_ar = min(ar_aic, key=ar_aic.get)
best_ma = min(ma_aic, key=ma_aic.get)

if ar_aic[best_ar] < ma_aic[best_ma]:
    best_model = f"AR({best_ar})"
    order = (best_ar, 0, 0)
else:
    best_model = f"MA({best_ma})"
    order = (0, 0, best_ma)

print("\nBest Model Selected:")
print(best_model)

final_model = ARIMA(series, order=order)
final_result = final_model.fit()

print(final_result.summary())

lb = acorr_ljungbox(final_result.resid, lags=[1], return_df=True)

print("\nLjung-Box Test Results:")
print(lb)
