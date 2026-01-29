# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Resources/ML471_S2_Datafile_Practice.csv")

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Use Close price only
close_series = df['Close'].dropna()

# =========================
# ADDITIVE DECOMPOSITION
# =========================
additive = seasonal_decompose(
    close_series,
    model='additive',
    period=12
)

# ---- ADDITIVE PLOTS ----
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

axes[0].plot(additive.observed)
axes[0].set_title("Observed")

axes[1].plot(additive.trend)
axes[1].set_title("Additive Decomposition: Trend")

axes[2].plot(additive.seasonal)
axes[2].set_title("Additive Decomposition: Seasonal")

axes[3].plot(additive.resid)
axes[3].set_title("Additive Decomposition: Residual")

plt.tight_layout()
plt.show()

# =========================
# MULTIPLICATIVE DECOMPOSITION
# =========================
multiplicative = seasonal_decompose(
    close_series,
    model='multiplicative',
    period=12
)

# ---- MULTIPLICATIVE PLOTS ----
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

axes[0].plot(multiplicative.observed)
axes[0].set_title("Observed")

axes[1].plot(multiplicative.trend)
axes[1].set_title("Multiplicative Decomposition: Trend")

axes[2].plot(multiplicative.seasonal)
axes[2].set_title("Multiplicative Decomposition: Seasonal")

axes[3].plot(multiplicative.resid)
axes[3].set_title("Multiplicative Decomposition: Residual")

plt.tight_layout()
plt.show()
