from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    confusion_matrix,
)

from src.feature_extractor import extract_dataset_features


def get_scores(model, path):
    df = pd.read_csv(path)

    X = extract_dataset_features(df["url"])
    X = X[list(model.feature_names_in_)]

    y = df["label"].to_numpy()
    scores = model.predict_proba(X)[:, 1]

    return y, scores


def expected_calibration_error(y, scores, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ids = np.minimum(
        np.searchsorted(edges[1:], scores, side="right"),
        bins - 1,
    )

    ece = 0.0

    for i in range(bins):
        mask = ids == i

        if not mask.any():
            continue

        ece += mask.mean() * abs(
            scores[mask].mean() - y[mask].mean()
        )

    return float(ece)


def evaluate(name, y, scores):
    predictions = (scores >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Method": name,
        "Brier": brier_score_loss(y, scores),
        "LogLoss": log_loss(y, scores, labels=[0, 1]),
        "ECE": expected_calibration_error(y, scores),
        "ROC_AUC": roc_auc_score(y, scores),
        "FP": int(fp),
        "FN": int(fn),
    }


def main():
    model = joblib.load(
        "models/diverse/random_forest_diverse.pkl"
    )
    assert list(model.classes_) == [0, 1]

    y_cal, raw_cal = get_scores(
        model, "data/compphish/validation.csv"
    )

    y_eval, raw_eval = get_scores(
        model, "data/diverse/phi_eval.csv"
    )

    # Sigmoid / Platt scaling
    sigmoid = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )
    sigmoid.fit(raw_cal.reshape(-1, 1), y_cal)

    sigmoid_scores = sigmoid.predict_proba(
        raw_eval.reshape(-1, 1)
    )[:, 1]

    # Isotonic calibration
    isotonic = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )
    isotonic.fit(raw_cal, y_cal)
    isotonic_scores = isotonic.predict(raw_eval)

    results = pd.DataFrame([
        evaluate("Uncalibrated", y_eval, raw_eval),
        evaluate("Sigmoid", y_eval, sigmoid_scores),
        evaluate("Isotonic", y_eval, isotonic_scores),
    ])

    print("\n=== CROSS-DATASET CALIBRATION ===")
    print(results.round(5).to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    results.to_csv(
        "results/cross_dataset_calibration.csv",
        index=False,
    )

    print("\nSaved: results/cross_dataset_calibration.csv")


if __name__ == "__main__":
    main()
