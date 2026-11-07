
import pandas as pd
from urllib.parse import urlparse

DATASETS = {
    "CompPhish Train": "data/compphish/train.csv",
    "CompPhish Validation": "data/compphish/validation.csv",
    "CompPhish Test": "data/compphish/test.csv",
    "PhiUSIIL Train": "data/diverse/phi_train.csv",
    "PhiUSIIL Evaluation": "data/diverse/phi_eval.csv",
}

for name, path in DATASETS.items():
    df = pd.read_csv(path)

    df["https"] = df["url"].astype(str).apply(
        lambda url: urlparse(url).scheme.lower() == "https"
    )

    print(f"\n=== {name} ===")
    print("Total URLs:", len(df))

    summary = df.groupby("label").agg(
        count=("https", "size"),
        https_percentage=("https", lambda x: x.mean() * 100)
    )

    print(summary.round(2).to_string())

print("\nHTTPS DATASET AUDIT COMPLETED")
