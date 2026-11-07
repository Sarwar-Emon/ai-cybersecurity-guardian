
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.feature_extractor import extract_dataset_features


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    print("Extracting URL features...")

    X_train = extract_dataset_features(train["url"])
    X_test = extract_dataset_features(test["url"])

    y_train = train["label"]
    y_test = test["label"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nRANDOM FOREST RESULTS")
    print("--------------------------")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print("F1:", f1_score(y_test, predictions))
    print("ROC-AUC:", roc_auc_score(y_test, probabilities))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nFeature Importances:")
    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print(importance.to_string())

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/random_forest.pkl")

    print("\nRandom Forest model saved!")


if __name__ == "__main__":
    main()
