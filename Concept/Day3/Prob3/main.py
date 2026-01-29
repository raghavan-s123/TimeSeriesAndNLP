import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("Resources/ML471_S2_Datafile_Concept(in).csv")

# Datetime handling
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# Target variable (already differenced)
series = df['Power_Consumption_diff'].dropna()

# ADF Test
adf_result = adfuller(series)
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

if adf_result[1] < 0.05:
    print("Series is stationary")
else:
    print("Series is NOT stationary")

# ==============================
# ACF & PACF SIDE BY SIDE
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

plot_acf(series, lags=30, ax=axes[0])
axes[0].set_title("ACF Plot")

plot_pacf(series, lags=30, ax=axes[1])
axes[1].set_title("PACF Plot")

plt.tight_layout()
plt.show()

# Train-test split
train_size = int(len(series) * 0.8)
train = series.iloc[:train_size]
test = series.iloc[train_size:]

# ARMA(1,1)
p, q = 1, 1
model = ARIMA(train, order=(p, 0, q))
model_fit = model.fit()

print(model_fit.summary())
print("AIC:", model_fit.aic)
print("BIC:", model_fit.bic)

# Ljung-Box Test
ljung_box = acorr_ljungbox(model_fit.resid, lags=[10], return_df=True)
print(ljung_box)

# Forecast
forecast = model_fit.forecast(steps=len(test))
forecast.index = test.index

# Forecast visualization
plt.figure(figsize=(12,5))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual Test Data')
plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
plt.title("ARMA Forecast vs Actual Power Consumption Changes")
plt.xlabel("Time")
plt.ylabel("Power_Consumption_diff")
plt.legend()
plt.show()

# Accuracy metrics
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test - forecast) / test)) * 100

print("MAE :", mae)
print("RMSE:", rmse)
print("MAPE:", mape)
