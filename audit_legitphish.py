
import pandas as pd

from src.feature_extractor import extract_dataset_features

# Load the original dataset
df = pd.read_csv("data/url_features_extracted1.csv")

print("ORIGINAL DATASET")
print("Total records:", len(df))

print("\nOriginal label distribution:")
print(df["ClassLabel"].value_counts(dropna=False))

# Remove missing labels
df = df.dropna(subset=["ClassLabel"]).copy()

# Check duplicate URLs
print("\nDuplicate URLs:", df.duplicated("URL").sum())

# Check conflicting labels
conflicts = (
    df.groupby("URL")["ClassLabel"]
    .nunique()
    .gt(1)
    .sum()
)

print("Conflicting URL labels:", conflicts)

# Extract features independently from raw URLs
print("\nExtracting URL features...")
X = extract_dataset_features(df["URL"])

audit = X[["https", "path_length", "url_length"]].copy()
audit["label"] = df["ClassLabel"].to_numpy()

print("\nHTTPS DISTRIBUTION (%)")
print(
    pd.crosstab(
        audit["label"],
        audit["https"],
        normalize="index"
    ).mul(100).round(2)
)

print("\nNONEMPTY PATH (%)")
print(
    audit.groupby("label")["path_length"]
    .apply(lambda x: (x > 0).mean() * 100)
    .round(2)
)

print("\nURL LENGTH STATISTICS")
print(
    audit.groupby("label")["url_length"]
    .describe()
    .to_string()
)
