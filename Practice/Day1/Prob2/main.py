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
# =========================
close_price = df['Close']

# =========================
# PLOT TIME SERIES
# =========================
plt.figure(figsize=(10, 5))
plt.plot(close_price, linewidth=1)

plt.title("Historical Stock Closing Price Over Time")
plt.xlabel("Year")
plt.ylabel("Closing Price")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
