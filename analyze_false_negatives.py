
import ipaddress
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import tldextract

from src.feature_extractor import extract_dataset_features


MODEL_PATH = "models/diverse/random_forest_diverse.pkl"
DATA_PATH = "data/url_features_extracted1.csv"

REFERENCE_FILES = [
    "data/compphish/train.csv",
    "data/compphish/validation.csv",
    "data/compphish/test.csv",
    "data/diverse/phi_train.csv",
]


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


# ==================================================
# 1. LOAD MODEL
# ==================================================

print("Loading Diverse Random Forest...")

model = joblib.load(MODEL_PATH)

if 1 not in model.classes_:
    raise ValueError("Phishing class 1 not found.")

phishing_class_index = list(model.classes_).index(1)


# ==================================================
# 2. BUILD DOMAIN EXCLUSION SET
# ==================================================

reference_domains = set()

for path in REFERENCE_FILES:
    reference_df = pd.read_csv(
        path, usecols=["url"]
    )

    reference_domains.update(
        reference_df["url"].apply(get_domain).dropna()
    )


# ==================================================
# 3. PREPARE LEGITPHISH
# ==================================================

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["URL", "ClassLabel"]
).copy()

df["url"] = df["URL"].astype(str).str.strip()

# Provisional mapping:
# LegitPhish 0 = phishing, 1 = legitimate
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

print("Eligible URLs:", len(df))


# ==================================================
# 4. EXTRACT FEATURES AND PREDICT
# ==================================================

X = extract_dataset_features(
    df["url"]
).reset_index(drop=True)

assert len(X) == len(df)

model_columns = list(model.feature_names_in_)

scores = model.predict_proba(
    X[model_columns]
)[:, phishing_class_index]

df["phishing_score"] = scores

df["prediction"] = (
    scores >= 0.50
).astype(int)

# Attach extracted features using positional arrays
for column in X.columns:
    df[column] = X[column].to_numpy()


# ==================================================
# 5. IDENTIFY TRUE POSITIVES AND FALSE NEGATIVES
# ==================================================

true_positives = df[
    (df["label"] == 1)
    & (df["prediction"] == 1)
].copy()

false_negatives = df[
    (df["label"] == 1)
    & (df["prediction"] == 0)
].copy()

print("\n=== PHISHING DETECTION ===")
print("Correctly detected:", len(true_positives))
print("Missed phishing:", len(false_negatives))

assert len(true_positives) + len(false_negatives) == (
    df["label"] == 1
).sum()

assert len(false_negatives) == 71, (
    "Expected 71 missed URLs. "
    "Check dataset filtering and model consistency."
)


# ==================================================
# 6. COMPARE FEATURE DISTRIBUTIONS
# ==================================================

features_to_compare = [
    "https",
    "url_length",
    "hostname_length",
    "path_length",
    "digit_count",
    "hyphen_count",
    "dot_count",
    "special_char_count",
    "query_length",
    "phishing_score",
]

print("\n=== FEATURE COMPARISON ===")

rows = []

for feature in features_to_compare:
    if feature not in df.columns:
        continue

    tp_values = true_positives[feature]
    fn_values = false_negatives[feature]

    rows.append({
        "Feature": feature,
        "Detected mean": round(tp_values.mean(), 4),
        "Missed mean": round(fn_values.mean(), 4),
        "Detected median": round(tp_values.median(), 4),
        "Missed median": round(fn_values.median(), 4),
    })

comparison = pd.DataFrame(rows)

print(comparison.to_string(index=False))


# ==================================================
# 7. MISSED PHISHING BY PROTOCOL
# ==================================================

print("\n=== MISSED PHISHING BY PROTOCOL ===")

for value, name in [(0, "HTTP"), (1, "HTTPS")]:
    detected = int(
        (true_positives["https"] == value).sum()
    )

    missed = int(
        (false_negatives["https"] == value).sum()
    )

    total = detected + missed

    print(f"\n{name}")
    print("Total phishing:", total)
    print("Detected:", detected)
    print("Missed:", missed)

    if total:
        print(
            "Miss rate:",
            round(missed / total * 100, 3),
            "%"
        )


# ==================================================
# 8. INSPECT MOST CONFIDENT MISTAKES
# ==================================================

print("\n=== MOST CONFIDENT FALSE NEGATIVES ===")

display_columns = [
    "url",
    "phishing_score",
    "https",
    "url_length",
    "path_length",
    "digit_count",
]

most_confident_errors = (
    false_negatives
    .sort_values("phishing_score")
    .head(10)
)

print(
    most_confident_errors[
        display_columns
    ].to_string(index=False)
)


# ==================================================
# 9. SAVE RESULTS
# ==================================================

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

false_negatives.to_csv(
    output_dir / "legitphish_false_negatives.csv",
    index=False
)

comparison.to_csv(
    output_dir / "false_negative_feature_comparison.csv",
    index=False
)

print("\nSaved:")
print("results/legitphish_false_negatives.csv")
print("results/false_negative_feature_comparison.csv")

print("\nSTEP 10.24 COMPLETED")
