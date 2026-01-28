from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Resources/ML471_S1_Datafile_Concept.csv")

df["DATE"] = pd.to_datetime(df["DATE"])
df.set_index("DATE", inplace=True)

df = df.dropna(subset=["Consumption"])

res = seasonal_decompose(df["Consumption"], model="additive", period=12)
res.plot()
plt.show()
