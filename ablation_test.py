
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

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

    "Without URL length": [
        c for c in X_train.columns
        if c != "url_length"
    ],

    "Without length features": [
        c for c in X_train.columns
        if c not in [
            "url_length",
            "hostname_length",
            "path_length",
            "query_length"
        ]
    ]
}

for name, columns in experiments.items():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X_train[columns], y_train)
    predictions = model.predict(X_test[columns])

    print(f"\n{name}")
    print("Features:", len(columns))
    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("Recall:", round(
        recall_score(y_test, predictions), 4
    ))
    print("F1:", round(
        f1_score(y_test, predictions), 4
    ))
