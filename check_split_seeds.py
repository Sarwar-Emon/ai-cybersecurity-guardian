
import pandas as pd
import tldextract
from urllib.parse import urlparse
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("data/compphish/clean_urls.csv")

def get_domain(url):
    hostname = (urlparse(url).hostname or "").lower()
    extracted = tldextract.extract(hostname)
    return extracted.registered_domain or hostname

df["domain"] = df["url"].apply(get_domain)

target = df["label"].mean()

results = []

for seed in range(20):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=seed
    )

    dev_idx, test_idx = next(
        splitter.split(df, df["label"], df["domain"])
    )

    dev = df.iloc[dev_idx]
    test = df.iloc[test_idx]

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.1765,
        random_state=seed + 100
    )

    train_idx, val_idx = next(
        splitter2.split(dev, dev["label"], dev["domain"])
    )

    train = dev.iloc[train_idx]
    val = dev.iloc[val_idx]

    rates = [
        train["label"].mean(),
        val["label"].mean(),
        test["label"].mean()
    ]

    score = sum(abs(rate - target) for rate in rates)

    results.append({
        "seed": seed,
        "train_n": len(train),
        "val_n": len(val),
        "test_n": len(test),
        "train_phishing": round(rates[0] * 100, 2),
        "val_phishing": round(rates[1] * 100, 2),
        "test_phishing": round(rates[2] * 100, 2),
        "imbalance_score": round(score, 4)
    })

result = pd.DataFrame(results)

print("\nOverall phishing rate:",
      round(target * 100, 2), "%")

print("\nTOP 5 CANDIDATE SPLITS")
print(
    result.sort_values("imbalance_score")
    .head(5)
    .to_string(index=False)
)
