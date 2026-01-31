import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Resources/ML471_S4_Datafile_Practice.csv")

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Ensure monthly frequency
df = df.asfreq("M")

# --- Seasonal Naïve using full history ---
season_length = 12
df["Seasonal_Naive_Forecast"] = df["Close"].shift(season_length)

# --- Keep ONLY 2014–2017 for plotting ---
df_plot = df.loc["2014-10-01":"2017-12-31"].dropna()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df_plot.index, df_plot["Close"], label="Actual Closing Price")
plt.plot(df_plot.index, df_plot["Seasonal_Naive_Forecast"],
         label="Seasonal Naïve Forecast")

plt.title("Actual vs Seasonal Naïve Forecast (2014–2017)")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
plt.show()
