
import pandas as pd
import tldextract

from sklearn.model_selection import GroupShuffleSplit

# Load cleaned dataset
df = pd.read_csv("data/phishing_urls_clean.csv")


def get_domain(url):
    extracted = tldextract.extract(str(url))

    domain = extracted.top_domain_under_public_suffix

    if domain:
        return domain

    # Keep unknown hostnames distinguishable
    return str(url).lower()


# Identify registrable domains
df["domain"] = df["url"].apply(get_domain)

print("Total records:", len(df))
print("Unique domain groups:", df["domain"].nunique())

# Split by domain, not by individual URL
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df,
        y=df["label"],
        groups=df["domain"]
    )
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

# Verify no domain overlap
overlap = (
    set(train_df["domain"])
    & set(test_df["domain"])
)

print("\nTraining records:", len(train_df))
print("Testing records:", len(test_df))
print("Overlapping domains:", len(overlap))

print("\nTraining class distribution:")
print(train_df["label"].value_counts(normalize=True))

print("\nTesting class distribution:")
print(test_df["label"].value_counts(normalize=True))

# Save split datasets
train_df[["url", "label"]].to_csv(
    "data/train.csv",
    index=False
)

test_df[["url", "label"]].to_csv(
    "data/test.csv",
    index=False
)

print("\nDomain-aware splitting completed!")
