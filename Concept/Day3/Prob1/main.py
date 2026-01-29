import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv("Resources/ML471_S2_Datafile_Concept(in).csv")
data['Date'] = pd.to_datetime(data['Datetime'])
data.set_index('Date', inplace=True)
data['Consumption_diff'] = data['Consumption'].diff()
data = data.dropna()

train_size = int(len(data) * 0.8)

train = data['Consumption_diff'][:train_size]
test = data['Consumption_diff'][train_size:]

arma_model = ARIMA(train, order=(2, 0, 2))
arma_result = arma_model.fit()

ljung_box = acorr_ljungbox(arma_result.resid, lags=[10], return_df=True)
print(ljung_box)

forecast = arma_result.forecast(steps=len(test))

plt.figure(figsize=(12,6))
plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Actual Test Data", linestyle="--")
plt.plot(test.index, forecast, label="Forecast", linestyle="--")

plt.title("ARMA(2,2) Forecast on Differenced Power Consumption")
plt.xlabel("Time")
plt.ylabel("Differenced Consumption")
plt.legend()
plt.grid()
plt.show()

residuals = arma_result.resid

plt.figure(figsize=(12,4))
plt.plot(residuals)
plt.title("Residual Time Series")
plt.show()