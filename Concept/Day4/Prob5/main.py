import pandas as pd
import os
import sys

file = input()

df = pd.read_csv(os.path.join(sys.path[0], file))

print("Dataset Preview:")
print(df.head())
print()

print("Dataset Information:")
print(df.info())
print()

if 'Datetime' in df.columns:
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df.set_index('Datetime', inplace=True)

print("Missing Value Check:")
print(df.isnull().sum())

df = df.dropna()

print("After missing value handling:")
print(df.isnull().sum())
print()

print("ACF and PACF Analysis:")
try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt

    plot_acf(df['Consumption'])
    plot_pacf(df['Consumption'])
    plt.close()

    print("ACF and PACF plots generated successfully.")
except Exception:
    print("Time series module not available. Skipping ACF/PACF plots.")
print()

split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

print("Train-Test Split:")
print(f"Training records: {len(train)}")
print(f"Testing records: {len(test)}")
