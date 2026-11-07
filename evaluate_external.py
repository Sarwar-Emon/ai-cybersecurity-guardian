
import ipaddress
from urllib.parse import urlparse

import joblib
import pandas as pd
import tldextract

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.feature_extractor import extract_dataset_features


# =====================================
# 1. LOAD MODEL AND DEVELOPMENT DATA
# =====================================

model = joblib.load(
    "models/compphish/random_forest.pkl"
)

train = pd.read_csv("data/compphish/train.csv")
val = pd.read_csv("data/compphish/validation.csv")

reference = pd.concat(
    [train, val],
    ignore_index=True
)


# =====================================
# 2. DOMAIN EXTRACTION
# =====================================

def get_domain(url):
    try:
        parsed = urlparse(str(url).strip())

        if parsed.scheme not in ("http", "https"):
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


reference_domains = set(
    reference["url"].apply(get_domain).dropna()
)


# =====================================
# 3. EXTERNAL EVALUATION FUNCTION
# =====================================

def evaluate(name, file_path, url_col, label_col, invert_labels=False):

    print(f"\n{'=' * 55}")
    print(f"EXTERNAL DATASET: {name}")
    print("=" * 55)

    df = pd.read_csv(file_path)

    df = df.dropna(
        subset=[url_col, label_col]
    ).copy()

    df["url"] = df[url_col].astype(str).str.strip()

    if invert_labels:
        df["label"] = 1 - df[label_col].astype(int)
    else:
        df["label"] = df[label_col].astype(int)

    assert set(df["label"].unique()) == {0, 1}

    df["domain"] = df["url"].apply(get_domain)

    # Exclude invalid and overlapping domains
    df = df[
        df["domain"].notna()
        & ~df["domain"].isin(reference_domains)
    ].copy()

    # Remove duplicate URLs
    df = df.drop_duplicates(subset=["url"])

    print("Eligible URLs:", len(df))

    print("\nClass distribution:")
    print(df["label"].value_counts())

    if len(df) == 0 or df["label"].nunique() < 2:
        print("Insufficient data for evaluation.")
        return

    # Extract the same 11 URL features
    print("\nExtracting URL features...")
    X = extract_dataset_features(df["url"])

    # Use the trained model without retraining
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    y = df["label"]

    print("\nRESULTS")
    print("Accuracy :", round(accuracy_score(y, predictions), 4))
    print("Precision:", round(precision_score(y, predictions), 4))
    print("Recall   :", round(recall_score(y, predictions), 4))
    print("F1-score :", round(f1_score(y, predictions), 4))
    print("ROC-AUC  :", round(roc_auc_score(y, probabilities), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))


# =====================================
# 4. EVALUATE PHIUSIIL
# =====================================

evaluate(
    name="PhiUSIIL",
    file_path="data/phishing_urls_clean.csv",
    url_col="url",
    label_col="label"
)


# =====================================
# 5. EVALUATE LEGITPHISH
# =====================================

evaluate(
    name="LegitPhish",
    file_path="data/url_features_extracted1.csv",
    url_col="URL",
    label_col="ClassLabel",
    invert_labels=True
)
