import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("Resources/ML471_S4_Datafile_Practice.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.dropna()

# -----------------------------
# 2. Prepare Prophet Data
# -----------------------------
prophet_df = df[['Date', 'Close', 'Volume']].copy()

prophet_df.rename(columns={
    'Date': 'ds',
    'Close': 'y',
    'Volume': 'volume'
}, inplace=True)

# -----------------------------
# 3. Train-Test Split
# -----------------------------
train_size = int(len(prophet_df) * 0.8)
train_df = prophet_df.iloc[:train_size]
test_df = prophet_df.iloc[train_size:]

# -----------------------------
# 4. Prophet Model
# -----------------------------
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

model.add_regressor('volume')

# -----------------------------
# 5. Fit Model
# -----------------------------
model.fit(train_df)

# -----------------------------
# 6. Future Data
# -----------------------------
future = model.make_future_dataframe(
    periods=len(test_df),
    freq='M'
)

# IMPORTANT: regressor required
future['volume'] = prophet_df['volume'].values

# -----------------------------
# 7. Forecast
# -----------------------------
forecast = model.predict(future)

# -----------------------------
# 8. EXACT Prophet Visualization (LIKE YOUR IMAGE)
# -----------------------------
fig = model.plot(forecast)
plt.show()

# -----------------------------
# 9. Components Plot (optional)
# -----------------------------
model.plot_components(forecast)
plt.show()
