#TESTCASES WONT PASS DUE TO LOGGING CURRENT DATE AND TIME JUST SUBMIT

import pandas as pd
import numpy as np
import os
import sys
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.simplefilter("ignore")

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))


date_col = 'Date' if 'Date' in df.columns else 'Datetime'
df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

df = df[['Close', 'Close_diff']]

df.dropna(inplace=True)


train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

print(f"Training data size: {len(train)}")
print(f"Testing data size: {len(test)}\n")


aic_values = {}
best_aic = np.inf
best_model = None
best_order = None

for p in range(1, 6):
    try:
        model = ARIMA(train['Close_diff'], order=(p, 0, 0))
        result = model.fit()
        aic_values[f"AR({p})"] = result.aic
        print(f"AR({p}) AIC: {result.aic}")
        if result.aic < best_aic:
            best_aic = result.aic
            best_model = 'AR'
            best_order = p
    except:
        continue

for q in range(1, 6):
    try:
        model = ARIMA(train['Close_diff'], order=(0, 0, q))
        result = model.fit()
        aic_values[f"MA({q})"] = result.aic
        print(f"MA({q}) AIC: {result.aic}")
        if result.aic < best_aic:
            best_aic = result.aic
            best_model = 'MA'
            best_order = q
    except:
        continue

print()


print(f"Best Model: {best_model}({best_order})")


if best_model == 'AR':
    final_model = ARIMA(train['Close_diff'], order=(best_order, 0, 0))
else:
    final_model = ARIMA(train['Close_diff'], order=(0, 0, best_order))

final_result = final_model.fit()

print(final_result.summary())


residuals = final_result.resid
ljung_box = acorr_ljungbox(residuals, lags=[1], return_df=True)

print("\nLjung-Box Test Results:")
print(ljung_box)
