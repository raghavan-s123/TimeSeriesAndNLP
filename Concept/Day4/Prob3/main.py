import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("Resources/ML471_S4_Datafile_Concept.csv")

# Convert Datetime column to datetime index
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# Keep only consumption column (adjust name if needed)
ts = df['Consumption']

# Handle missing values
ts = ts.dropna()

train_size = int(len(ts) * 0.8)
train = ts.iloc[:train_size]
test = ts.iloc[train_size:]

model = auto_arima(
    train,
    seasonal=True,
    m=12,                  # yearly seasonality for monthly data
    start_p=0,
    start_q=0,
    max_p=3,
    max_q=3,
    start_P=0,
    start_Q=0,
    max_P=2,
    max_Q=2,
    d=None,
    D=1,                   # seasonal differencing
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print(model.summary())


n_periods = len(test)

forecast = model.predict(n_periods=n_periods)
forecast = pd.Series(forecast, index=test.index)

plt.figure(figsize=(12, 6))

plt.plot(train, label="Training Data", color="blue")
plt.plot(test, label="Actual Test Data", linestyle="--", color="orange")
plt.plot(forecast, label="SARIMA Forecast", linestyle="--", color="green")

plt.title("SARIMA Forecast for Monthly Power Consumption")
plt.xlabel("Date")
plt.ylabel("Power Consumption")
plt.legend()
plt.grid(True)

plt.show()
