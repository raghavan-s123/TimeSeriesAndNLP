import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Resources/ML471_S2_Datafile_Concept(in).csv")

# Convert Datetime column and set index
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# =========================
# USE DIFFERENCED SERIES
# =========================
power_diff = df['Power_Consumption_diff'].dropna()

# =========================
# ADF TEST (NO EMOJIS)
# =========================
adf_result = adfuller(power_diff)

print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

if adf_result[1] < 0.05:
    print("Series is stationary after differencing")
else:
    print("Series is NOT stationary")

# =========================
# ACF & PACF PLOTS (SIDE BY SIDE)
# =========================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ACF
plot_acf(power_diff, lags=30, ax=axes[0])
axes[0].set_title("ACF")
axes[0].set_xlabel("Lags")
axes[0].set_ylabel("ACF Values")

# PACF
plot_pacf(power_diff, lags=30, ax=axes[1], method='ywm')
axes[1].set_title("PACF")
axes[1].set_xlabel("Lags")
axes[1].set_ylabel("PACF Values")

plt.tight_layout()
plt.show()
