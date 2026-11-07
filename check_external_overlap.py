
import ipaddress
from urllib.parse import urlparse

import pandas as pd
import tldextract


# Load datasets
comp_train = pd.read_csv("data/compphish/train.csv")
comp_val = pd.read_csv("data/compphish/validation.csv")

phiusiil = pd.read_csv("data/phishing_urls_clean.csv")
legitphish = pd.read_csv("data/url_features_extracted1.csv")

print("PhiUSIIL columns:", phiusiil.columns.tolist())
print("LegitPhish columns:", legitphish.columns.tolist())


# Domain extraction
def get_domain(url):
    try:
        hostname = (urlparse(str(url)).hostname or "").lower()

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


# Reference domains from model development
reference = pd.concat(
    [comp_train, comp_val],
    ignore_index=True
)

reference_urls = set(reference["url"].astype(str))
reference_domains = set(
    reference["url"].apply(get_domain).dropna()
)


def audit_dataset(name, df, url_column):
    urls = df[url_column].dropna().astype(str)

    domains = urls.apply(get_domain)

    exact_overlap = urls.isin(reference_urls)
    domain_overlap = domains.isin(reference_domains)
    invalid = domains.isna()

    eligible = (
        ~exact_overlap
        & ~domain_overlap
        & ~invalid
    )

    print(f"\n{'=' * 45}")
    print(name)
    print("=" * 45)

    print("Total records:", len(df))
    print("Exact URL overlap:", int(exact_overlap.sum()))
    print("Domain overlap:", int(domain_overlap.sum()))
    print("Invalid URL/domain:", int(invalid.sum()))
    print("Eligible external records:", int(eligible.sum()))

    print(
        "Eligible percentage:",
        round(eligible.mean() * 100, 2),
        "%"
    )


# PhiUSIIL: detect URL column
phi_url_candidates = [
    "url", "URL", "Url"
]

phi_url = next(
    (c for c in phi_url_candidates if c in phiusiil.columns),
    None
)

if phi_url:
    audit_dataset("PhiUSIIL", phiusiil, phi_url)
else:
    print("\nPhiUSIIL URL column not identified.")


# LegitPhish
audit_dataset(
    "LegitPhish",
    legitphish,
    "URL"
)
