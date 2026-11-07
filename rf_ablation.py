
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.feature_extractor import extract_dataset_features

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print("Extracting features...")

X_train = extract_dataset_features(train["url"])
X_test = extract_dataset_features(test["url"])

y_train = train["label"]
y_test = test["label"]

experiments = {
    "All features": list(X_train.columns),

    "Without HTTPS and path": [
        c for c in X_train.columns
        if c not in ["https", "path_length"]
    ],

    "Without HTTPS, path, and URL length": [
        c for c in X_train.columns
        if c not in [
            "https",
            "path_length",
            "url_length"
        ]
    ]
}

for name, columns in experiments.items():
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train[columns], y_train)
    predictions = model.predict(X_test[columns])

    print(f"\n{name}")
    print("Features:", len(columns))
    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("Precision:", round(
        precision_score(y_test, predictions), 4
    ))
    print("Recall:", round(
        recall_score(y_test, predictions), 4
    ))
    print("F1:", round(
        f1_score(y_test, predictions), 4
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
