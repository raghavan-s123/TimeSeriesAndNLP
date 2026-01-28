from statsmodels.tsa.stattools import acf
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Resources/ML471_S1_Datafile_Concept.csv")

x = df['Consumption'].values
acf_vals, conf = acf(x, nlags=365, alpha=0.05)

lower = conf[:, 0] - acf_vals
upper = conf[:, 1] - acf_vals

plt.plot(range(len(acf_vals)), acf_vals)
plt.axhline(0, color='black')
plt.axhline(0.1, linestyle='--', color='black')
plt.axhline(-0.1, linestyle='--', color='black')

plt.show()