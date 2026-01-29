import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("Resources/ML471_S2_Datafile_Concept(in).csv")
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)
data = data.dropna()

# =========================
# TRAIN–TEST SPLIT
# =========================
train_size = int(len(data) * 0.8)
train = data['Consumption'].iloc[:train_size]
test = data['Consumption'].iloc[train_size:]

# =========================
# ARIMA MODEL (2,1,2)
# =========================
model = ARIMA(train, order=(2, 1, 2))
model_fit = model.fit()

# =========================
# LJUNG–BOX TEST (IN-SAMPLE)
# =========================
ljung_box = acorr_ljungbox(model_fit.resid, lags=[10], return_df=True)
print("Ljung–Box Test on Training Residuals:")
print(ljung_box)

# =========================
# FORECAST
# =========================
forecast = model_fit.forecast(steps=len(test))
forecast.index = test.index  # IMPORTANT

# =========================
# ERROR METRICS
# =========================
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print(f"\nMAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# =========================
# PLOT TRAIN / TEST / FORECAST
# =========================
plt.figure(figsize=(12, 6))
plt.plot(train, label="Training Data")
plt.plot(test, label="Actual Test Data", linestyle="--")
plt.plot(forecast, label="Forecast", linestyle="--")
plt.title("ARIMA(2,1,2) Forecast on Power Consumption")
plt.xlabel("Time")
plt.ylabel("Power Consumption")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# FORECAST RESIDUALS (CORRECT)
# =========================
test_residuals = test - forecast

plt.figure(figsize=(12, 4))
plt.plot(test_residuals, color='blue')
plt.axhline(0, linestyle='--', color='black')
plt.title("Forecast Residuals (Test − Forecast)")
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()
