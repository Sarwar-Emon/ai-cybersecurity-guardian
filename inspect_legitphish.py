
from pathlib import Path
import pandas as pd

data_folder = Path("data")

# Find CSV files in the data folder
csv_files = list(data_folder.glob("*.csv"))

print("Available CSV files:")
for file in csv_files:
    print("-", file.name)

# Replace this with the downloaded CSV filename
filename = "url_features_extracted1.csv"

df = pd.read_csv(data_folder / filename)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

print("\nMissing values:")
print(df.isnull().sum().to_string())
