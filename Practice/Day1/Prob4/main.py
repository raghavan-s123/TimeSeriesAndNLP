import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Resources/ML471_S1_Datafile_Practice.csv")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# =========================
# RESAMPLE TO MONTHLY DATA
# =========================
# Use monthly mean to match sample visualization
monthly_close = df['Close'].resample('M').mean()

# =========================
# ADDITIVE DECOMPOSITION
# =========================
additive_decomp = seasonal_decompose(
    monthly_close,
    model='additive',
    period=12
)

plt.figure(figsize=(10, 8))
additive_decomp.plot()
plt.suptitle("Additive Decomposition", fontsize=14)
plt.tight_layout()
plt.show()

# =========================
# MULTIPLICATIVE DECOMPOSITION
# =========================
multiplicative_decomp = seasonal_decompose(
    monthly_close,
    model='multiplicative',
    period=12
)

plt.figure(figsize=(10, 8))
multiplicative_decomp.plot()
plt.suptitle("Multiplicative Decomposition", fontsize=14)
plt.tight_layout()
plt.show()
