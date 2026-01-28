import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Resources/ML471_S1_Datafile_Concept.csv")

data["DATE"] = pd.to_datetime(data["DATE"])

plt.figure(figsize=(12, 6))
plt.plot(
    data["DATE"],
    data["Consumption"],
    label="Electricity Consumption"
)

plt.title("Electricity Consumption Over Time")
plt.xlabel("Date")
plt.ylabel("Consumption")
plt.legend()

plt.show()
