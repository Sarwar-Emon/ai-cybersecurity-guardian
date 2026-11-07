
import ipaddress
from urllib.parse import urlparse

import pandas as pd
import tldextract


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


# Load training sources
comp_train = pd.read_csv("data/compphish/train.csv")
phi_train = pd.read_csv("data/diverse/phi_train.csv")

training = pd.concat(
    [comp_train[["url"]], phi_train[["url"]]],
    ignore_index=True
)

training_domains = set(
    training["url"].apply(get_domain).dropna()
)

training_urls = set(
    training["url"].astype(str).str.strip()
)

print("Training reference domains:", len(training_domains))


# Load LegitPhish
df = pd.read_csv("data/url_features_extracted1.csv")

print("\n=== LEGITPHISH OVERLAP AUDIT ===")
print("Original records:", len(df))

df = df.dropna(subset=["URL", "ClassLabel"]).copy()

df["url"] = df["URL"].astype(str).str.strip()

# Check conflicting labels
conflicts = (
    df.groupby("url")["ClassLabel"]
    .nunique()
    .gt(1)
    .sum()
)

print("Conflicting labels:", conflicts)

if conflicts:
    raise ValueError("Conflicting URL labels detected.")

df = df.drop_duplicates(subset=["url"])

print("Records after deduplication:", len(df))

df["domain"] = df["url"].apply(get_domain)

invalid = df["domain"].isna()
exact_overlap = df["url"].isin(training_urls)
domain_overlap = df["domain"].isin(training_domains)

print("Invalid URLs:", int(invalid.sum()))
print("Exact URL overlaps:", int(exact_overlap.sum()))
print("Domain overlaps:", int(domain_overlap.sum()))

eligible = df[
    ~invalid
    & ~exact_overlap
    & ~domain_overlap
].copy()

print("\nEligible external URLs:", len(eligible))
print("Eligible unique domains:", eligible["domain"].nunique())

print("\nOriginal ClassLabel distribution:")
print(eligible["ClassLabel"].value_counts())

print(
    "\nEligible percentage:",
    f"{len(eligible) / len(df) * 100:.2f}%"
)

print("\nAUDIT COMPLETED")
