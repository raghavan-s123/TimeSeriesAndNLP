import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

df.head()

plt.figure(figsize=(10, 4))
plt.plot(df['Date'], df['Close'], label='Consumption over Time')
