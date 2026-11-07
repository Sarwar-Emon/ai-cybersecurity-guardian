
from pathlib import Path
import pandas as pd

source = Path("data/compphish/Mapping_File.xlsx")
destination = Path("data/compphish/clean_urls.csv")

df = pd.read_excel(source)

print("Original records:", len(df))

# Remove missing URL or label values
df = df.dropna(subset=["url", "label"]).copy()

# Standardize URL strings
df["url"] = df["url"].astype(str).str.strip()

# Remove empty URLs
df = df[df["url"] != ""].copy()

# Verify labels
assert set(df["label"].unique()) == {0, 1}, (
    "Unexpected label values found"
)

# Check for conflicting labels before deduplication
conflicts = (
    df.groupby("url")["label"]
    .nunique()
    .gt(1)
    .sum()
)

if conflicts:
    raise ValueError(
        f"Found {conflicts} URLs with conflicting labels"
    )

# Remove exact duplicate URLs
df = df.drop_duplicates(subset=["url"]).copy()

# Retain only the columns needed for our model
df = df[["url", "label"]]

destination.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(destination, index=False)

print("Cleaned records:", len(df))
print("Removed:", 15358 - len(df))

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nSaved to:", destination)
