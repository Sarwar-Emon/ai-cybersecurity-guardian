
import pandas as pd

# Load original dataset
df = pd.read_csv("data/phishing_urls.csv")

print("Original records:", len(df))

# Check for conflicting labels
conflicts = (
    df.groupby("url")["label"]
    .nunique()
    .gt(1)
    .sum()
)

if conflicts > 0:
    raise ValueError(
        f"Found {conflicts} URLs with conflicting labels"
    )

# Remove exact duplicate URLs
df_clean = df.drop_duplicates(subset=["url"]).copy()

print("Cleaned records:", len(df_clean))
print("Removed:", len(df) - len(df_clean))

# Save cleaned dataset
df_clean.to_csv(
    "data/phishing_urls_clean.csv",
    index=False
)

print("\nCleaned class distribution:")
print(df_clean["label"].value_counts())

print("\nDataset cleaning completed!")
