
import pandas as pd

df = pd.read_csv("data/phishing_urls.csv")

print("Total records:", len(df))

# Check exact duplicate URLs
duplicates = df.duplicated(subset=["url"]).sum()
print("Duplicate URLs:", duplicates)

# Check conflicting labels for identical URLs
conflicts = (
    df.groupby("url")["label"]
    .nunique()
    .gt(1)
    .sum()
)
print("URLs with conflicting labels:", conflicts)

# Check class balance
print("\nClass distribution (%):")
print(
    df["label"].value_counts(normalize=True) * 100
)

# Analyze URL length
df["url_length"] = df["url"].str.len()

print("\nURL length statistics:")
print(df.groupby("label")["url_length"].describe())
