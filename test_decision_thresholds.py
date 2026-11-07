from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

from src.feature_extractor import extract_dataset_features


MODEL_PATH = "models/diverse/random_forest_diverse.pkl"
THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

DATASETS = {
    "CompPhish Validation": "data/compphish/validation.csv",
    "PhiUSIIL Evaluation": "data/diverse/phi_eval.csv",
}


def main():
    model = joblib.load(MODEL_PATH)

    assert list(model.classes_) == [0, 1]

    results = []

    for dataset_name, path in DATASETS.items():
        df = pd.read_csv(path)
        X = extract_dataset_features(df["url"])
        X = X[list(model.feature_names_in_)]

        y_true = df["label"].to_numpy()
        scores = model.predict_proba(X)[:, 1]

        print(f"\n=== {dataset_name} ===")
        print("Records:", len(df))

        for threshold in THRESHOLDS:
            y_pred = (scores >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).ravel()

            results.append({
                "Dataset": dataset_name,
                "Threshold": threshold,
                "Precision": precision_score(
                    y_true, y_pred, zero_division=0
                ),
                "Recall": recall_score(
                    y_true, y_pred, zero_division=0
                ),
                "F1": f1_score(
                    y_true, y_pred, zero_division=0
                ),
                "Balanced Accuracy": balanced_accuracy_score(
                    y_true, y_pred
                ),
                "FPR": fp / (fp + tn),
                "FP": int(fp),
                "FN": int(fn),
            })

    results_df = pd.DataFrame(results)

    print("\n=== THRESHOLD COMPARISON ===")
    print(results_df.round(4).to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    results_df.to_csv(
        "results/decision_threshold_comparison.csv",
        index=False
    )

    print("\nSaved: results/decision_threshold_comparison.csv")


if __name__ == "__main__":
    main()
