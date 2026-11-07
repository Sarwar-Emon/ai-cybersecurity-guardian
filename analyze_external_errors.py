
import ipaddress
from urllib.parse import urlparse

import joblib
import pandas as pd
import tldextract

from src.feature_extractor import extract_dataset_features


model = joblib.load(
    "models/compphish/random_forest.pkl"
)

train = pd.read_csv("data/compphish/train.csv")
val = pd.read_csv("data/compphish/validation.csv")


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


reference = pd.concat(
    [train, val],
    ignore_index=True
)

reference_domains = set(
    reference["url"].apply(get_domain).dropna()
)


# Load PhiUSIIL
df = pd.read_csv("data/phishing_urls_clean.csv")

df["domain"] = df["url"].apply(get_domain)

df = df[
    df["domain"].notna()
    & ~df["domain"].isin(reference_domains)
].copy()

df = df.drop_duplicates(subset=["url"])

# Extract features
X = extract_dataset_features(df["url"])

df["prediction"] = model.predict(X)

# False negatives: phishing classified as legitimate
false_negatives = df[
    (df["label"] == 1)
    & (df["prediction"] == 0)
].copy()

# Correctly detected phishing
true_positives = df[
    (df["label"] == 1)
    & (df["prediction"] == 1)
].copy()

print("False negatives:", len(false_negatives))
print("True positives:", len(true_positives))

# Compare URL characteristics
for name, subset in [
    ("MISSED PHISHING", false_negatives),
    ("DETECTED PHISHING", true_positives)
]:
    features = extract_dataset_features(subset["url"])

    print(f"\n{name}")
    print("Average URL length:", round(features["url_length"].mean(), 2))
    print("Average hostname length:", round(features["hostname_length"].mean(), 2))
    print("Average path length:", round(features["path_length"].mean(), 2))
    print("HTTPS percentage:", round(features["https"].mean() * 100, 2))
    print("Average hyphen count:", round(features["hyphen_count"].mean(), 2))
    print("Average digit count:", round(features["digit_count"].mean(), 2))

print("\nAnalysis complete!")
