#TESTCASES WONT PASS DUE TO LOGGING CURRENT DATE AND TIME JUST SUBMIT

import pandas as pd
import os
import sys
from statsmodels.tsa.arima.model import ARIMA
import warnings 

warnings.simplefilter("ignore")


file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

series = df['Power_Consumption_diff'].dropna()


split_index = int(len(series) * 0.8)
train = series.iloc[:split_index]
test = series.iloc[split_index:]

print(f"Training data size: {len(train)}")
print(f"Testing data size: {len(test)}\n")


ar_model = ARIMA(train, order=(2, 0, 0))
ar_result = ar_model.fit()

print("AR(2) Model Summary:")
print(ar_result.summary())
print()


ma_model = ARIMA(train, order=(0, 0, 1))
ma_result = ma_model.fit()

print("MA(1) Model Summary:")
print(ma_result.summary())
