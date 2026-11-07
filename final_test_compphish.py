
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
)

from src.feature_extractor import extract_dataset_features


# ==========================================
# 1. LOAD RESERVED TEST SET
# ==========================================

test = pd.read_csv("data/compphish/test.csv")

assert test["url"].notna().all()
assert test["label"].notna().all()
assert set(test["label"].unique()) == {0, 1}
assert not test["url"].duplicated().any()

print("FINAL COMPPhish TEST")
print("=" * 50)
print("Test URLs:", len(test))
print("Legitimate:", (test["label"] == 0).sum())
print("Phishing:", (test["label"] == 1).sum())


# ==========================================
# 2. EXTRACT SAME 11 FEATURES
# ==========================================

print("\nExtracting test features...")

X_test = extract_dataset_features(test["url"])
y_test = test["label"]

assert not X_test.isnull().any().any()


# ==========================================
# 3. LOAD SAVED MODELS
# ==========================================

models = {
    "Baseline RF": joblib.load(
        "models/compphish/random_forest.pkl"
    ),
    "Diverse RF": joblib.load(
        "models/diverse/random_forest_diverse.pkl"
    ),
}


# ==========================================
# 4. FINAL EVALUATION
# ==========================================

THRESHOLD = 0.50
results = []

for name, model in models.items():

    # Verify expected feature names.
    expected_features = list(model.feature_names_in_)

    assert X_test.columns.tolist() == expected_features, (
        f"Feature mismatch for {name}"
    )

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        ),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
        "True Negatives": int(tn),
    }

    results.append(metrics)

    print(f"\n{'=' * 50}")
    print(name)
    print("=" * 50)

    for metric in [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]:
        print(f"{metric}: {metrics[metric]:.4f}")

    print("\nConfusion Matrix:")
    print([[int(tn), int(fp)], [int(fn), int(tp)]])

    print("\nFalse positives:", fp)
    print("False negatives:", fn)


# ==========================================
# 5. SAVE FINAL RESULTS
# ==========================================

output = Path("models/final_evaluation")
output.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    output / "compphish_final_test_results.csv",
    index=False
)

print("\nFINAL TEST COMPARISON")
print("=" * 50)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nFinal evaluation completed.")
print("Do not tune models using these test results.")
