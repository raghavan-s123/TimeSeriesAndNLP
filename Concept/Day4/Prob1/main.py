import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_absolute_error, mean_squared_error


data = pd.read_csv("Resources/ML471_S4_Datafile_Concept.csv")

data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)

data = data[['Consumption', 'Festivals/Special_events']]

data.dropna(inplace=True)


train_size = int(len(data) * 0.8)

train = data.iloc[:train_size]
test  = data.iloc[train_size:]

y_train = train['Consumption']
y_test  = test['Consumption']

exog_train = train[['Festivals/Special_events']]
exog_test  = test[['Festivals/Special_events']]


print("\nGranger Causality Test Results")
print("-" * 40)

granger_data = data[['Consumption', 'Festivals/Special_events']]

grangercausalitytests(
    granger_data,
    maxlag=12,
    verbose=True
)


model = SARIMAX(
    y_train,
    exog=exog_train,
    order=(1, 0, 2),
    seasonal_order=(0, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(disp=False)

print(model_fit.summary())


forecast = model_fit.get_forecast(
    steps=len(test),
    exog=exog_test
)

forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()


plt.figure(figsize=(12, 6))

plt.plot(train.index, y_train, label='Training Data', color='blue')
plt.plot(test.index, y_test, label='Actual Test Data', color='orange', linestyle='--')
plt.plot(test.index, forecast_mean, label='SARIMAX Forecast', color='green', linestyle='--')


plt.title("SARIMAX Forecast with External Variable")
plt.xlabel("Date")
plt.ylabel("Power Consumption")
plt.legend()
plt.grid(True)
plt.show()


mae = mean_absolute_error(y_test, forecast_mean)
rmse = np.sqrt(mean_squared_error(y_test, forecast_mean))
mape = np.mean(np.abs((y_test - forecast_mean) / y_test)) * 100

print("\nModel Performance Metrics")
print("-" * 40)
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"MAPE : {mape:.2f}%")
