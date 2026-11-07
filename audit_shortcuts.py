
import pandas as pd

from src.feature_extractor import extract_dataset_features

df = pd.read_csv("data/test.csv")

X = extract_dataset_features(df["url"])

audit = X[["https", "path_length", "url_length"]].copy()
audit["label"] = df["label"].to_numpy()

print("\nHTTPS DISTRIBUTION BY CLASS (%)")
print(
    pd.crosstab(
        audit["label"],
        audit["https"],
        normalize="index"
    ).mul(100).round(2)
)

print("\nPATH LENGTH STATISTICS BY CLASS")
print(
    audit.groupby("label")["path_length"]
    .describe()
    .to_string()
)

print("\nPERCENTAGE WITH EMPTY PATH")
print(
    audit.groupby("label")["path_length"]
    .apply(lambda x: (x == 0).mean() * 100)
    .round(2)
)

print("\nPERCENTAGE WITH URL LENGTH > 58")
print(
    audit.groupby("label")["url_length"]
    .apply(lambda x: (x > 58).mean() * 100)
    .round(2)
)
