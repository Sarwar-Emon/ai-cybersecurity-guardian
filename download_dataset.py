
from pathlib import Path
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Create data directory if needed
Path("data").mkdir(exist_ok=True)

# Download PhiUSIIL dataset from UCI
dataset = fetch_ucirepo(id=967)

# Extract original URLs and labels
urls = dataset.data.features["URL"]
labels = dataset.data.targets["label"]

# Create our project dataset
df = pd.DataFrame({
    "url": urls,
    "label": (labels == 0).astype(int)
})

# Save locally
df.to_csv("data/phishing_urls.csv", index=False)

print("Dataset downloaded successfully!")
print("Shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())
print("\nMissing values:")
print(df.isnull().sum())
print("\nFirst 5 records:")
print(df.head())
