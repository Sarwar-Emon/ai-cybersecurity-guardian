
import pandas as pd

from src.feature_extractor import extract_dataset_features

# Load CompPhish
df = pd.read_excel(
    "data/compphish/Mapping_File.xlsx"
)

print("TOTAL RECORDS:", len(df))

print("\nLABEL DISTRIBUTION")
print(df["label"].value_counts(dropna=False))

print("\nDUPLICATE URLS")
print(df.duplicated(subset=["url"]).sum())

print("\nCONFLICTING LABELS")
conflicts = (
    df.groupby("url")["label"]
    .nunique()
    .gt(1)
    .sum()
)
print(conflicts)

print("\nEXTRACTING FEATURES...")
X = extract_dataset_features(df["url"])

audit = X[
    ["https", "path_length", "url_length"]
].copy()

audit["label"] = df["label"].to_numpy()

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
