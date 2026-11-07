
import joblib
import pandas as pd

from urllib.parse import urlsplit, urlunsplit
from sklearn.ensemble import RandomForestClassifier

from src.feature_extractor import extract_dataset_features


# Load deployed model
full_model = joblib.load(
    "models/diverse/random_forest_diverse.pkl"
)

# Reconstruct original training data
comp = pd.read_csv("data/compphish/train.csv")
phi = pd.read_csv("data/diverse/phi_train.csv")

phi_0 = phi[phi["label"] == 0].sample(
    n=5000, random_state=42
)
phi_1 = phi[phi["label"] == 1].sample(
    n=5000, random_state=42
)

train = pd.concat(
    [comp, phi_0, phi_1],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

X_train = extract_dataset_features(train["url"])

# Train model without HTTPS
no_https_columns = [
    c for c in X_train.columns if c != "https"
]

no_https_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

no_https_model.fit(
    X_train[no_https_columns],
    train["label"]
)

# Evaluation URLs
test = pd.read_csv("data/compphish/test.csv")

# Use URLs with standard HTTP or HTTPS schemes
urls = []

for url in test["url"].astype(str):
    parsed = urlsplit(url)
    if parsed.scheme.lower() in ("http", "https"):
        urls.append(url)

# Limit experiment size for quick testing
#urls = urls[:500]


def switch_protocol(url):
    parsed = urlsplit(url)
    new_scheme = (
        "http" if parsed.scheme.lower() == "https"
        else "https"
    )

    return urlunsplit((
        new_scheme,
        parsed.netloc,
        parsed.path,
        parsed.query,
        parsed.fragment
    ))


changed_urls = [switch_protocol(url) for url in urls]

X_original = extract_dataset_features(urls)
X_changed = extract_dataset_features(changed_urls)


def evaluate_robustness(model, columns, name):
    original_scores = model.predict_proba(
        X_original[columns]
    )[:, 1]

    changed_scores = model.predict_proba(
        X_changed[columns]
    )[:, 1]

    original_labels = original_scores >= 0.5
    changed_labels = changed_scores >= 0.5

    flip_rate = (
        original_labels != changed_labels
    ).mean() * 100

    mean_change = abs(
        original_scores - changed_scores
    ).mean() * 100

    print(f"\n{name}")
    print(f"URLs tested: {len(urls)}")
    print(f"Classification flip rate: {flip_rate:.2f}%")
    print(
        "Mean absolute score change:",
        f"{mean_change:.2f} percentage points"
    )


evaluate_robustness(
    full_model,
    list(full_model.feature_names_in_),
    "Original Random Forest"
)

evaluate_robustness(
    no_https_model,
    no_https_columns,
    "Random Forest Without HTTPS"
)

print("\nPROTOCOL ROBUSTNESS TEST COMPLETED")
