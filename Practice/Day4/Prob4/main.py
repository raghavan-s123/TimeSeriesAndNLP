import os
import sys
import pandas as pd

file = input().strip()
df = pd.read_csv(os.path.join(sys.path[0], file))


print("Dataset Preview:")
print(df.head())
print()


print("Dataset Information:")
info = df.info()
print(info)
print()


print("Missing Value Check:")
base_cols = ["Open", "High", "Low", "Close", "Volume", "Close_diff"]
print(df[base_cols].isnull().sum())
print()


df = df.dropna(subset=["Close"])

df["Close_diff"] = df["Close_diff"].fillna(method="ffill")

df = df.dropna(subset=["Close_diff"])

print("After missing value handling:")
print(df[base_cols].isnull().sum())
print()

if "Date" not in df.columns:
    print("Date column missing. Cannot proceed.")
    sys.exit()

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)


y = df["Close"].dropna()


split = int(len(y) * 0.8)
train = y.iloc[:split]
test = y.iloc[split:]

print("Train-Test Split:")
print(f"Training records: {len(train)}")
print(f"Testing records: {len(test)}")
print()


print("SARIMA Model Summary:")

try:
    from pmdarima import auto_arima

    model = auto_arima(
        train,
        seasonal=True,
        m=12,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )

    print(model.summary())

except Exception:
    print("pmdarima not available. SARIMA modeling skipped.")

