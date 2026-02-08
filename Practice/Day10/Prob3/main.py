import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("Resources/ML471_S4_Datafile_Practice.csv")   # Change filename if needed

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values('Date')

# Remove missing values
df = df.dropna()

# -----------------------------
# 2. Prepare Data for Prophet
# -----------------------------
# Keep only required columns
prophet_df = df[['Date', 'Close', 'Volume']].copy()

# Rename columns
prophet_df.rename(columns={
    'Date': 'ds',
    'Close': 'y',
    'Volume': 'volume'
}, inplace=True)

# -----------------------------
# 3. Train-Test Split (80-20)
# -----------------------------
train_size = int(len(prophet_df) * 0.8)

train_df = prophet_df.iloc[:train_size]
test_df = prophet_df.iloc[train_size:]

# -----------------------------
# 4. Initialize Prophet Model
# -----------------------------
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

# Add Volume as External Regressor
model.add_regressor('volume')

# -----------------------------
# 5. Train Model
# -----------------------------
model.fit(train_df)

# -----------------------------
# 6. Create Future Dataframe
# -----------------------------
future = model.make_future_dataframe(
    periods=len(test_df),
    freq='M'   # Monthly data
)

# Add volume values (mandatory)
future['volume'] = prophet_df['volume'].values

# -----------------------------
# 7. Generate Forecast
# -----------------------------
forecast_prophet = model.predict(future)

# -----------------------------
# 8. Main Forecast Plot
# -----------------------------
model.plot(forecast_prophet)
plt.title("Stock Price Forecast using Prophet")
plt.show()

# -----------------------------
# 9. Component Decomposition Plot
# -----------------------------
model.plot_components(forecast_prophet)
plt.show()
