import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv("Resources/ML471_S3_Datafile_Practice.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df.drop(columns=["SMA_10","SMA_30","SES"])
series = df["Close_diff"].dropna()

fig, ax = plt.subplots(1, 2, figsize=(12,5))

plot_acf(series, lags=30, ax=ax[0])
ax[0].set_title("ACF")
ax[0].set_xlabel("Lags")
ax[0].set_ylabel("ACF Values")

plot_pacf(series, lags=30, ax=ax[1], method="ywm", color='red')
ax[1].set_title("PACF")
ax[1].set_xlabel("Lags")
ax[1].set_ylabel("PACF Values")

plt.tight_layout()
plt.show()