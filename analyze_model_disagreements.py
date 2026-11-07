
import ipaddress
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import tldextract

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from src.feature_extractor import extract_dataset_features


def get_domain(url):
    try:
        parsed = urlparse(str(url).strip())

        if parsed.scheme.lower() not in ("http", "https"):
            return None

        hostname = (parsed.hostname or "").lower()

        if not hostname:
            return None

        try:
            ipaddress.ip_address(hostname)
            return hostname
        except ValueError:
            pass

        extracted = tldextract.extract(hostname)

        return (
            extracted.top_domain_under_public_suffix
            or hostname
        )
    except (ValueError, TypeError):
        return None


# STEP 1: Reconstruct the diverse training sample
print("Preparing training data...")

comp_train = pd.read_csv("data/compphish/train.csv")
phi_train = pd.read_csv("data/diverse/phi_train.csv")

phi_0 = phi_train[phi_train["label"] == 0].sample(
    n=5000, random_state=42
)
phi_1 = phi_train[phi_train["label"] == 1].sample(
    n=5000, random_state=42
)

train = pd.concat(
    [comp_train, phi_0, phi_1],
    ignore_index=True
).sample(
    frac=1, random_state=42
).reset_index(drop=True)

X_train = extract_dataset_features(train["url"])

# STEP 2: Train no-HTTPS candidate
print("Training no-HTTPS Random Forest...")

no_https_columns = [
    c for c in X_train.columns if c != "https"
]

no_https_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

no_https_model.fit(
    X_train[no_https_columns],
    train["label"]
)

# STEP 3: Load deployed Diverse RF
diverse_model = joblib.load(
    "models/diverse/random_forest_diverse.pkl"
)

# STEP 4: Exclude domains from model development
reference_files = [
    "data/compphish/train.csv",
    "data/compphish/validation.csv",
    "data/compphish/test.csv",
    "data/diverse/phi_train.csv",
]

reference_domains = set()

for path in reference_files:
    reference_df = pd.read_csv(
        path, usecols=["url"]
    )

    reference_domains.update(
        reference_df["url"].apply(get_domain).dropna()
    )

# STEP 5: Prepare external LegitPhish records
df = pd.read_csv(
    "data/url_features_extracted1.csv"
)

df = df.dropna(
    subset=["URL", "ClassLabel"]
).copy()

df["url"] = df["URL"].astype(str).str.strip()

# Provisional mapping:
# Original 0 = phishing, original 1 = legitimate
df["label"] = 1 - df["ClassLabel"].astype(int)

conflicts = df.groupby("url")["label"].nunique()
bad_urls = set(conflicts[conflicts > 1].index)

df = df[~df["url"].isin(bad_urls)]
df = df.drop_duplicates(subset=["url"])

df["domain"] = df["url"].apply(get_domain)

df = df[
    df["domain"].notna()
    & ~df["domain"].isin(reference_domains)
].copy().reset_index(drop=True)

print("Eligible external URLs:", len(df))

# STEP 6: Predict using both models
X = extract_dataset_features(df["url"])

diverse_columns = list(
    diverse_model.feature_names_in_
)

no_https_columns = list(
    no_https_model.feature_names_in_
)

df["diverse_score"] = diverse_model.predict_proba(
    X[diverse_columns]
)[:, 1]

df["no_https_score"] = no_https_model.predict_proba(
    X[no_https_columns]
)[:, 1]

df["diverse_pred"] = (
    df["diverse_score"] >= 0.5
).astype(int)

df["no_https_pred"] = (
    df["no_https_score"] >= 0.5
).astype(int)

df["diverse_correct"] = (
    df["diverse_pred"] == df["label"]
)

df["no_https_correct"] = (
    df["no_https_pred"] == df["label"]
)

df["disagree"] = (
    df["diverse_pred"] != df["no_https_pred"]
)

# STEP 7: Summarize disagreements
print("\n=== MODEL DISAGREEMENTS ===")

print("Total disagreements:", int(df["disagree"].sum()))

print(
    "Disagreement rate:",
    round(df["disagree"].mean() * 100, 2),
    "%"
)

diverse_only = (
    df["diverse_correct"]
    & ~df["no_https_correct"]
)

no_https_only = (
    df["no_https_correct"]
    & ~df["diverse_correct"]
)

print("Only Diverse RF correct:", int(diverse_only.sum()))
print("Only No-HTTPS RF correct:", int(no_https_only.sum()))

print("\n=== DISAGREEMENTS BY TRUE CLASS ===")

print(
    pd.crosstab(
        df.loc[df["disagree"], "label"],
        df.loc[df["disagree"], "diverse_pred"],
        rownames=["True label"],
        colnames=["Diverse prediction"]
    )
)

# STEP 8: Compare URL characteristics
df["https"] = X["https"].to_numpy()
df["url_length"] = X["url_length"].to_numpy()
df["path_length"] = X["path_length"].to_numpy()

print("\n=== FEATURE PROFILES ===")

groups = {
    "Both correct": (
        df["diverse_correct"]
        & df["no_https_correct"]
    ),
    "Only Diverse correct": diverse_only,
    "Only No-HTTPS correct": no_https_only,
    "Both wrong": (
        ~df["diverse_correct"]
        & ~df["no_https_correct"]
    ),
}

for name, mask in groups.items():
    subset = df.loc[mask]

    print(f"\n{name}: {len(subset)} URLs")

    if subset.empty:
        continue

    print(
        "HTTPS percentage:",
        round(subset["https"].mean() * 100, 2)
    )
    print(
        "Mean URL length:",
        round(subset["url_length"].mean(), 2)
    )
    print(
        "Mean path length:",
        round(subset["path_length"].mean(), 2)
    )

# STEP 9: Save results for manual inspection
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

columns_to_save = [
    "url",
    "domain",
    "label",
    "https",
    "url_length",
    "path_length",
    "diverse_score",
    "no_https_score",
    "diverse_pred",
    "no_https_pred",
    "diverse_correct",
    "no_https_correct",
]

df.loc[
    df["disagree"], columns_to_save
].to_csv(
    output_dir / "legitphish_disagreements.csv",
    index=False
)

print(
    "\nSaved:",
    output_dir / "legitphish_disagreements.csv"
)

print("\nSTEP 10.20 COMPLETED")
