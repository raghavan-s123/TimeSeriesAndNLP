import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv("Resources/ML471_S1_Datafile_Concept.csv")

df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

consumption = df['Consumption']


diff_series = consumption.diff().dropna()

plt.figure()
plt.plot(diff_series)
plt.title("Differenced Electricity Consumption")
plt.xlabel("Time")
plt.ylabel("Differenced Value")

ax = plt.gca()

major_years = range(1989, 2020, 5)
ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in major_years])
ax.set_xticklabels(major_years)

ax.xaxis.set_minor_locator(mdates.YearLocator(1))

ax.tick_params(axis='x', which='minor', length=4)
ax.tick_params(axis='x', which='major', length=8)

plt.show()


adf_result = adfuller(diff_series)

print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

if adf_result[1] < 0.05:
    print("✅ Series is stationary after differencing")
else:
    print("❌ Series is NOT stationary")

model = ARIMA(consumption, order=(1, 1, 1))
model_fit = model.fit()

print(model_fit.summary())

forecast = model_fit.forecast(steps=10)

plt.figure()
plt.plot(consumption, label="Original")
plt.plot(forecast, label="Forecast", color="red")
plt.legend()
plt.title("Electricity Consumption Forecast")

ax = plt.gca()


major_years = range(1989, 2020, 5)
ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in major_years])
ax.set_xticklabels(major_years)

ax.xaxis.set_minor_locator(mdates.YearLocator(1))

ax.tick_params(axis='x', which='minor', length=4)
ax.tick_params(axis='x', which='major', length=8)

plt.show()
