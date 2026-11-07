
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


# Load training domain references
comp_train = pd.read_csv("data/compphish/train.csv")
comp_val = pd.read_csv("data/compphish/validation.csv")

# Read only URLs from reserved CompPhish test.
comp_test = pd.read_csv(
    "data/compphish/test.csv",
    usecols=["url"]
)

phi_train = pd.read_csv("data/diverse/phi_train.csv")

reference = pd.concat(
    [
        comp_train[["url"]],
        comp_val[["url"]],
        comp_test[["url"]],
        phi_train[["url"]]
    ],
    ignore_index=True
)

reference_domains = set(
    reference["url"].apply(get_domain).dropna()
)


# Load external LegitPhish
df = pd.read_csv("data/url_features_extracted1.csv")

df = df.dropna(
    subset=["URL", "ClassLabel"]
).copy()

df["url"] = df["URL"].astype(str).str.strip()

# LegitPhish: 0=phishing, 1=legitimate
df["label"] = 1 - df["ClassLabel"].astype(int)

df["domain"] = df["url"].apply(get_domain)

df = df[
    df["domain"].notna()
    & ~df["domain"].isin(reference_domains)
].copy()

conflicts = (
    df.groupby("url")["label"]
    .nunique()
    .gt(1)
    .sum()
)

print("Conflicting URL labels:", conflicts)

# Exclude URLs with conflicting labels
if conflicts:
    bad_urls = (
        df.groupby("url")["label"]
        .nunique()
    )
    bad_urls = set(bad_urls[bad_urls > 1].index)
    df = df[~df["url"].isin(bad_urls)]

df = df.drop_duplicates(subset=["url"])

print("Eligible LegitPhish URLs:", len(df))
print("Class distribution:")
print(df["label"].value_counts())


# Extract features once
X = extract_dataset_features(df["url"])
y = df["label"]

assert y.nunique() == 2


models = {
    "Baseline RF": joblib.load(
        "models/compphish/random_forest.pkl"
    ),
    "Diverse RF": joblib.load(
        "models/diverse/random_forest_diverse.pkl"
    ),
}


for name, model in models.items():
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    print(f"\n{'=' * 45}")
    print(name)
    print("=" * 45)

    print("Accuracy:",
          round(accuracy_score(y, predictions), 4))
    print("Precision:",
          round(precision_score(y, predictions), 4))
    print("Recall:",
          round(recall_score(y, predictions), 4))
    print("F1:",
          round(f1_score(y, predictions), 4))
    print("ROC-AUC:",
          round(roc_auc_score(y, probabilities), 4))

    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))
