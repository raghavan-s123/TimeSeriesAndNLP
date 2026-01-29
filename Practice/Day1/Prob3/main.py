import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA 
# =========================
df = pd.read_csv("Resources/ML471_S1_Datafile_Practice.csv")
# Convert Date column to datetime and set index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
# =========================
# SELECT CLOSE PRICE
close_price = df['Close']
# =========================

# =========================
# CALCULATE MOVING AVERAGES
sma_10 = close_price.rolling(window=10).mean()
sma_30 = close_price.rolling(window=30).mean()
# =========================
# PLOT SMA
plt.figure(figsize=(12, 6))
plt.plot(close_price, label='Close Price', color='blue', linewidth=1)
plt.plot(sma_10, label='10-Day SMA', color='orange', linewidth=1)
plt.plot(sma_30, label='30-Day SMA', color='green', linewidth=1)
plt.title("Close Price with 10-Day and 30-Day Simple Moving Averages")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# =========================
# SIMPLE EXPONENTIAL SMOOTHING (SES)
alpha = 0.2  # Smoothing factor
ses = close_price.ewm(alpha=alpha, adjust=False).mean()
# =========================
# PLOT SES
plt.figure(figsize=(12, 6))
plt.plot(close_price, label='Close Price', color='blue', linewidth=1)
plt.plot(ses, label='SES', color='red', linewidth=1)
plt.title("Close Price with Simple Exponential Smoothing (SES)")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
