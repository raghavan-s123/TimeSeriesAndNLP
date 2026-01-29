import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("Resources/ML471_S3_Datafile_Practice.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
series = df["Close_diff"].dropna()

split = int(len(series) * 0.8)
train = series.iloc[:split]
test = series.iloc[split:]

model = ARIMA(train, order=(1,0,1))
fitted = model.fit()

forecast = fitted.forecast(steps=len(test))
forecast.index = test.index

plt.figure(figsize=(10,5))
plt.plot(train.index, train, color="blue", label="Train")
plt.plot(test.index, test, color="orange", linestyle="--", label="Actual")
plt.plot(forecast.index, forecast, color="green", linestyle="--", label="Forecast")
plt.title("ARMA(1,0,1) Forecast Plot")
plt.legend()
plt.show()

residuals = test - forecast

plt.figure(figsize=(10,4))
plt.plot(residuals.index, residuals)
plt.title("Residuals")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.show()