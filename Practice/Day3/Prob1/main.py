import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("Resources/ML471_S3_Datafile_Practice.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
series = df['Close']

train = series[:'2014-12']
test = series['2015-01':]

model = ARIMA(train, order=(1,1,1))
fit = model.fit()
forecast = fit.forecast(steps=len(test))

residuals = test - forecast

plt.figure(figsize=(10,5))
plt.plot(train, color='blue', label='Train')
plt.plot(test, color='orange', linestyle='--', label='Actual')
plt.plot(forecast, color='green', linestyle='--', label='Forecast')
plt.title("ARIMA(1,1,1) Forecast Plot")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(residuals)
plt.title("Residuals")
plt.ylabel("Residuals")
plt.xlabel("Date")
plt.show()