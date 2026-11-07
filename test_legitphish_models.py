
import ipaddress
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import tldextract

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.feature_extractor import extract_dataset_features


# ============================================================
# 1. DOMAIN EXTRACTION
# ============================================================

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


# ============================================================
# 2. RECONSTRUCT DIVERSE TRAINING DATA
# ============================================================

print("Preparing diverse training data...")

comp_train = pd.read_csv(
    "data/compphish/train.csv"
)

phi_train = pd.read_csv(
    "data/diverse/phi_train.csv"
)

phi_0 = phi_train[
    phi_train["label"] == 0
].sample(
    n=5000,
    random_state=42
)

phi_1 = phi_train[
    phi_train["label"] == 1
].sample(
    n=5000,
    random_state=42
)

train = pd.concat(
    [comp_train, phi_0, phi_1],
    ignore_index=True
)

train = train.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_train = extract_dataset_features(
    train["url"]
).reset_index(drop=True)

y_train = train["label"].to_numpy()

assert len(X_train) == len(y_train)

columns_no_https = [
    column
    for column in X_train.columns
    if column != "https"
]


# ============================================================
# 3. TRAIN NO-HTTPS RANDOM FOREST
# ============================================================

print("Training no-HTTPS candidate...")

no_https = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

no_https.fit(
    X_train[columns_no_https],
    y_train
)


# ============================================================
# 4. LOAD EXISTING MODELS
# ============================================================

models = {
    "Baseline RF": joblib.load(
        "models/compphish/random_forest.pkl"
    ),

    "Diverse RF": joblib.load(
        "models/diverse/random_forest_diverse.pkl"
    ),

    "No-HTTPS RF": no_https,
}


# ============================================================
# 5. COLLECT REFERENCE DOMAINS
# ============================================================

reference_files = [
    "data/compphish/train.csv",
    "data/compphish/validation.csv",
    "data/compphish/test.csv",
    "data/diverse/phi_train.csv",
]

reference_domains = set()

for path in reference_files:

    reference_df = pd.read_csv(
        path,
        usecols=["url"]
    )

    domains = reference_df[
        "url"
    ].apply(get_domain).dropna()

    reference_domains.update(domains)

print(
    "Reference domains:",
    len(reference_domains)
)


# ============================================================
# 6. LOAD AND CLEAN LEGITPHISH
# ============================================================

print("\nPreparing LegitPhish dataset...")

df = pd.read_csv(
    "data/url_features_extracted1.csv"
)

original_count = len(df)

df = df.dropna(
    subset=["URL", "ClassLabel"]
).copy()

df["url"] = (
    df["URL"]
    .astype(str)
    .str.strip()
)

# IMPORTANT:
# Provisional label mapping.
# Must be verified using original dataset documentation.
#
# Original ClassLabel 0 = Phishing
# Original ClassLabel 1 = Legitimate
#
# Model convention:
# 1 = Phishing
# 0 = Legitimate

assert df["ClassLabel"].isin([0, 1]).all()

df["label"] = (
    1 - df["ClassLabel"].astype(int)
)

# Identify conflicting labels
conflicts = (
    df.groupby("url")["label"]
    .nunique()
)

bad_urls = set(
    conflicts[conflicts > 1].index
)

print(
    "Conflicting URL labels:",
    len(bad_urls)
)

df = df[
    ~df["url"].isin(bad_urls)
].copy()

# Remove duplicate URLs
df = df.drop_duplicates(
    subset=["url"]
).copy()

# Extract domains
df["domain"] = df["url"].apply(
    get_domain
)

# Remove invalid and overlapping domains
df = df[
    df["domain"].notna()
    & ~df["domain"].isin(reference_domains)
].copy()

# CRITICAL: Reset index to maintain row alignment
df = df.reset_index(drop=True)

print("\nOriginal LegitPhish records:", original_count)
print("Eligible LegitPhish URLs:", len(df))
print("Unique eligible domains:", df["domain"].nunique())

print("\nClass distribution:")
print(df["label"].value_counts())

if df["label"].nunique() != 2:
    raise ValueError(
        "Both classes are required for evaluation."
    )


# ============================================================
# 7. EXTRACT FEATURES AND VERIFY ALIGNMENT
# ============================================================

X = extract_dataset_features(
    df["url"]
).reset_index(drop=True)

y = df["label"].to_numpy()

assert len(X) == len(y) == len(df)
assert X.index.equals(df.index)

print("\nFeature extraction completed.")
print("Evaluation samples:", len(X))


# ============================================================
# 8. OVERALL MODEL EVALUATION
# ============================================================

print("\n" + "=" * 55)
print("OVERALL MODEL COMPARISON")
print("=" * 55)

predictions_by_model = {}

for name, model in models.items():

    columns = list(
        model.feature_names_in_
    )

    missing_features = [
        column
        for column in columns
        if column not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"{name} missing features: {missing_features}"
        )

    X_model = X[columns]

    predictions = model.predict(
        X_model
    )

    class_one_index = list(
        model.classes_
    ).index(1)

    scores = model.predict_proba(
        X_model
    )[:, class_one_index]

    predictions_by_model[name] = predictions

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    ).ravel()

    print(f"\n=== {name} ===")

    print(
        "Accuracy:",
        round(accuracy_score(y, predictions), 4)
    )

    print(
        "Balanced Accuracy:",
        round(balanced_accuracy_score(y, predictions), 4)
    )

    print(
        "Precision:",
        round(precision_score(y, predictions), 4)
    )

    print(
        "Recall:",
        round(recall_score(y, predictions), 4)
    )

    print(
        "F1:",
        round(f1_score(y, predictions), 4)
    )

    print(
        "ROC-AUC:",
        round(roc_auc_score(y, scores), 4)
    )

    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

    print(
        "False Positive Rate:",
        round(fp / (fp + tn) * 100, 2),
        "%"
    )

    print(
        "False Negative Rate:",
        round(fn / (fn + tp) * 100, 2),
        "%"
    )


# ============================================================
# 9. VERIFIED HTTP / HTTPS PHISHING RECALL
# ============================================================

print("\n" + "=" * 55)
print("VERIFIED PROTOCOL-SPECIFIC PHISHING RECALL")
print("=" * 55)

protocols = X["https"].to_numpy()

assert len(protocols) == len(y)

assert np.isin(protocols, [0, 1]).all()

for name, predictions in predictions_by_model.items():

    print(f"\n=== {name} ===")

    overall_missed = int(
        (
            (y == 1)
            & (predictions == 0)
        ).sum()
    )

    subgroup_missed = 0
    subgroup_total = 0

    for protocol_value, protocol_name in [
        (0, "HTTP"),
        (1, "HTTPS"),
    ]:

        mask = (
            (y == 1)
            & (protocols == protocol_value)
        )

        total = int(mask.sum())

        missed = int(
            (predictions[mask] == 0).sum()
        )

        subgroup_missed += missed
        subgroup_total += total

        if total > 0:
            recall = 1 - missed / total
        else:
            recall = float("nan")

        print(f"\n{protocol_name} phishing URLs:", total)
        print("Missed phishing:", missed)
        print("Phishing recall:", round(recall, 4))

    print("\nOverall missed phishing:", overall_missed)
    print("Sum of subgroup misses:", subgroup_missed)

    assert subgroup_missed == overall_missed, (
        f"{name}: subgroup false negatives "
        "do not match overall false negatives."
    )

    assert subgroup_total == int((y == 1).sum()), (
        f"{name}: protocol groups do not cover "
        "all phishing records."
    )

    print("Consistency check: PASSED")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("LEGITPHISH EVALUATION COMPLETED")
print("ALL CONSISTENCY CHECKS PASSED")
print("=" * 55)
