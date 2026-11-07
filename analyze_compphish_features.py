
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.feature_extractor import extract_dataset_features


# Load training and validation data
train = pd.read_csv("data/compphish/train.csv")
val = pd.read_csv("data/compphish/validation.csv")

X_train = extract_dataset_features(train["url"])
X_val = extract_dataset_features(val["url"])

y_train = train["label"]
y_val = val["label"]

# Load trained Random Forest
model = joblib.load(
    "models/compphish/random_forest.pkl"
)

print("\nRANDOM FOREST FEATURE IMPORTANCE")

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance.to_string(index=False))


# =====================================
# ABLATION STUDY
# =====================================

from sklearn.ensemble import RandomForestClassifier

experiments = {
    "All features": [],
    "Without HTTPS and path": [
        "https",
        "path_length"
    ],
    "Without HTTPS, path, URL length": [
        "https",
        "path_length",
        "url_length"
    ]
}

for name, removed in experiments.items():

    features = [
        c for c in X_train.columns
        if c not in removed
    ]

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_train[features],
        y_train
    )

    predictions = rf.predict(X_val[features])

    print(f"\n{name}")
    print("Accuracy:",
          round(accuracy_score(y_val, predictions), 4))
    print("Precision:",
          round(precision_score(y_val, predictions), 4))
    print("Recall:",
          round(recall_score(y_val, predictions), 4))
    print("F1:",
          round(f1_score(y_val, predictions), 4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, predictions))
