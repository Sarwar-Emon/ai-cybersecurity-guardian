
import ipaddress
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit


# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("data/compphish/clean_urls.csv")

assert df["url"].notna().all(), "Missing URLs found"
assert set(df["label"].unique()) == {0, 1}, "Invalid labels"


# ==========================================
# 2. EXTRACT REGISTRABLE DOMAINS
# ==========================================

def get_domain(url):
    hostname = (urlparse(str(url)).hostname or "").lower()

    if not hostname:
        raise ValueError(f"Missing hostname: {url}")

    # Handle IP-based URLs
    try:
        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        pass

    extracted = tldextract.extract(hostname)

    domain = extracted.top_domain_under_public_suffix

    # Fallback for unknown public suffixes
    return domain or hostname


df["domain"] = df["url"].apply(get_domain)

print("Total URLs:", len(df))
print("Total unique domains:", df["domain"].nunique())


# ==========================================
# 3. DEVELOPMENT / TEST SPLIT
# ==========================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=16
)

dev_idx, test_idx = next(
    splitter.split(
        df,
        df["label"],
        groups=df["domain"]
    )
)

dev = df.iloc[dev_idx].copy()
test = df.iloc[test_idx].copy()


# ==========================================
# 4. TRAIN / VALIDATION SPLIT
# ==========================================

splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.1765,
    random_state=116
)

train_idx, val_idx = next(
    splitter2.split(
        dev,
        dev["label"],
        groups=dev["domain"]
    )
)

train = dev.iloc[train_idx].copy()
val = dev.iloc[val_idx].copy()


# ==========================================
# 5. REPORT SPLIT STATISTICS
# ==========================================

def report(name, data):
    print(f"\n{name}")
    print("Records:", len(data))
    print("Unique domains:", data["domain"].nunique())

    print("Class distribution:")
    print(
        data["label"]
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
    )


report("TRAIN", train)
report("VALIDATION", val)
report("TEST", test)


# ==========================================
# 6. VERIFY DATA INTEGRITY
# ==========================================

train_domains = set(train["domain"])
val_domains = set(val["domain"])
test_domains = set(test["domain"])

assert train_domains.isdisjoint(val_domains)
assert train_domains.isdisjoint(test_domains)
assert val_domains.isdisjoint(test_domains)

print("\nDomain overlap across splits: 0")

for name, data in [
    ("train", train),
    ("validation", val),
    ("test", test)
]:
    assert set(data["label"]) == {0, 1}, (
        f"{name} must contain both classes"
    )

# Check that all records are accounted for
assert len(train) + len(val) + len(test) == len(df)

# Check that no URLs overlap
assert set(train["url"]).isdisjoint(val["url"])
assert set(train["url"]).isdisjoint(test["url"])
assert set(val["url"]).isdisjoint(test["url"])

print("URL overlap across splits: 0")
print("All integrity checks passed!")


# ==========================================
# 7. SAVE DATASETS
# ==========================================

output = Path("data/compphish")
output.mkdir(parents=True, exist_ok=True)

train[["url", "label"]].to_csv(
    output / "train.csv",
    index=False
)

val[["url", "label"]].to_csv(
    output / "validation.csv",
    index=False
)

test[["url", "label"]].to_csv(
    output / "test.csv",
    index=False
)

print("\nSplit files saved successfully!")
print("Training:", output / "train.csv")
print("Validation:", output / "validation.csv")
print("Testing:", output / "test.csv")
