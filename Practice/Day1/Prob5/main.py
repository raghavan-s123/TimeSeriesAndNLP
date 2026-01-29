import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load data
df = pd.read_csv("Resources/ML471_S1_Datafile_Practice.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# =========================
# ADDITIVE (MONTHLY MEAN)
# =========================
monthly_close_add = df['Close'].resample('M').mean()

add = seasonal_decompose(monthly_close_add, model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Additive Decomposition")

axes[0].plot(add.observed); axes[0].set_title("Close")
axes[1].plot(add.trend);    axes[1].set_title("Trend")
axes[2].plot(add.seasonal); axes[2].set_title("Seasonal")
axes[3].plot(add.resid);    axes[3].set_title("Residual")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# =========================
# MULTIPLICATIVE (MONTHLY LAST)
# =========================
monthly_close_mul = df['Close'].resample('M').mean()

mul = seasonal_decompose(monthly_close_mul, model='multiplicative', period=12)

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Multiplicative Decomposition")

axes[0].plot(mul.observed); axes[0].set_title("Close")
axes[1].plot(mul.trend);    axes[1].set_title("Trend")
axes[2].plot(mul.seasonal); axes[2].set_title("Seasonal")
axes[3].plot(mul.resid);    axes[3].set_title("Residual")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
