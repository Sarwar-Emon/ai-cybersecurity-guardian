
from pathlib import Path
from urllib.parse import urlparse
import ipaddress

import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit


OUTPUT = Path("data/diverse")
OUTPUT.mkdir(parents=True, exist_ok=True)


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


# Load CompPhish partitions.
comp_train = pd.read_csv("data/compphish/train.csv")
comp_val = pd.read_csv("data/compphish/validation.csv")
comp_test = pd.read_csv(
    "data/compphish/test.csv",
    usecols=["url"]
)

comp_all = pd.concat(
    [
        comp_train[["url"]],
        comp_val[["url"]],
        comp_test[["url"]]
    ],
    ignore_index=True
)

comp_domains = set(
    comp_all["url"].apply(get_domain).dropna()
)


# Load PhiUSIIL.
phi = pd.read_csv("data/phishing_urls_clean.csv")

phi = phi[["url", "label"]].dropna().copy()
phi["url"] = phi["url"].astype(str).str.strip()
phi["label"] = phi["label"].astype(int)

assert set(phi["label"].unique()) == {0, 1}

phi["domain"] = phi["url"].apply(get_domain)

# Exclude invalid and overlapping domains.
phi = phi[
    phi["domain"].notna()
    & ~phi["domain"].isin(comp_domains)
].copy()

# Check conflicting labels before deduplication.
conflicts = (
    phi.groupby("url")["label"]
    .nunique()
    .gt(1)
    .sum()
)

assert conflicts == 0, "Conflicting labels found!"

phi = phi.drop_duplicates(subset=["url"])

print("Eligible PhiUSIIL URLs:", len(phi))
print("Eligible domains:", phi["domain"].nunique())


# Split by registrable domain.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, eval_idx = next(
    splitter.split(
        phi,
        phi["label"],
        groups=phi["domain"]
    )
)

phi_train = phi.iloc[train_idx].copy()
phi_eval = phi.iloc[eval_idx].copy()


# Verify isolation.
train_domains = set(phi_train["domain"])
eval_domains = set(phi_eval["domain"])

assert train_domains.isdisjoint(eval_domains)
assert train_domains.isdisjoint(comp_domains)
assert eval_domains.isdisjoint(comp_domains)

assert phi_train["label"].nunique() == 2
assert phi_eval["label"].nunique() == 2


# Save only required columns.
phi_train[["url", "label"]].to_csv(
    OUTPUT / "phi_train.csv",
    index=False
)

phi_eval[["url", "label"]].to_csv(
    OUTPUT / "phi_eval.csv",
    index=False
)


print("\nPHIUSIIL DEVELOPMENT SPLIT")
print("=" * 45)

for name, df in [
    ("Training candidates", phi_train),
    ("Held-out evaluation", phi_eval)
]:
    print(f"\n{name}")
    print("URLs:", len(df))
    print("Unique domains:", df["domain"].nunique())
    print("Class distribution:")
    print(df["label"].value_counts())
    print("Phishing percentage:",
          round(df["label"].mean() * 100, 2))


print("\nDomain overlap:", len(train_domains & eval_domains))
print("CompPhish domain overlap:",
      len((train_domains | eval_domains) & comp_domains))

print("\nFiles saved successfully.")
