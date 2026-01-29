# ============================================================
# Exponential Smoothing Analysis of Electricity Consumption
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    ExponentialSmoothing
)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("Resources/ML471_S1_Datafile_Concept.csv")

df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

consumption = df['Consumption']


# ------------------------------------------------------------
# 3. SINGLE EXPONENTIAL SMOOTHING (SES)
# ------------------------------------------------------------
ses_model = SimpleExpSmoothing(consumption).fit(
    smoothing_level=0.2,
    optimized=False
)
ses_fitted = ses_model.fittedvalues



# ------------------------------------------------------------
# 4. DOUBLE EXPONENTIAL SMOOTHING (HOLT'S METHOD)
# ------------------------------------------------------------
holt_model = ExponentialSmoothing(
    consumption,
    trend='additive',
    seasonal=None
).fit()

holt_fitted = holt_model.fittedvalues



# ------------------------------------------------------------
# 5. TRIPLE EXPONENTIAL SMOOTHING (HOLT-WINTERS)
# ------------------------------------------------------------
hw_model = ExponentialSmoothing(
    consumption,
    trend='additive',
    seasonal='additive',
    seasonal_periods=12
).fit()

hw_fitted = hw_model.fittedvalues



# ------------------------------------------------------------
# 6. COMPARISON OF ALL METHODS
# ------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(consumption, label="Original Data", alpha=0.5)
plt.plot(ses_fitted, label="SES")
plt.plot(holt_fitted, label="Holt")
plt.plot(hw_fitted, label="Holt-Winters")
plt.title("Comparison of Exponential Smoothing Techniques")
plt.xlabel("Date")
plt.ylabel("Consumption")
plt.legend()
plt.grid(True)
plt.show()
