import pandas as pd
import os
import sys

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

print("First 5 rows of the dataset:")
print(df.head())
print()

print("Missing values in dataset:")
print(df.drop(columns='Date').isnull().sum())
print()


dupli = df.duplicated().sum()
print(f"Number of duplicate rows: {dupli}")
print()

print("Close price summary statistics:")
print(df['Close'].describe())