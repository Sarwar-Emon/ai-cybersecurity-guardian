
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.feature_extractor import extract_dataset_features


def main():
    # Load the training and testing datasets
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    print("Extracting URL features...")

    X_train = extract_dataset_features(train_df["url"])
    X_test = extract_dataset_features(test_df["url"])

    y_train = train_df["label"]
    y_test = test_df["label"]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Scaling is fitted on training data only
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nMODEL EVALUATION")
    print("------------------------------")

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"]
    ))

    # Save the trained model
    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        model,
        "models/logistic_regression.pkl"
    )

    print("\nModel saved successfully!")


if __name__ == "__main__":
    main()
