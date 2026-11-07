from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.feature_extractor import extract_dataset_features


DATASETS = {
    "CompPhish Validation": "data/compphish/validation.csv",
    "PhiUSIIL Evaluation": "data/diverse/phi_eval.csv",
}

N_BINS = 10


def calibration_table(y_true, scores):
    edges = np.linspace(0, 1, N_BINS + 1)

    # Bin 0 covers [0, 0.1); final bin includes 1.0.
    bin_ids = np.minimum(
        np.searchsorted(edges[1:], scores, side="right"),
        N_BINS - 1,
    )

    rows = []

    for i in range(N_BINS):
        mask = bin_ids == i
        count = int(mask.sum())

        if count == 0:
            continue

        mean_score = float(scores[mask].mean())
        observed_rate = float(y_true[mask].mean())
        gap = abs(mean_score - observed_rate)

        rows.append({
            "Bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}",
            "Count": count,
            "Mean Score": mean_score,
            "Observed Phishing Rate": observed_rate,
            "Absolute Gap": gap,
        })

    table = pd.DataFrame(rows)

    ece = float(
        (
            table["Count"] / len(y_true)
            * table["Absolute Gap"]
        ).sum()
    )

    return table, ece


def main():
    model = joblib.load(
        "models/diverse/random_forest_diverse.pkl"
    )

    assert list(model.classes_) == [0, 1]

    Path("results").mkdir(exist_ok=True)

    for name, path in DATASETS.items():
        df = pd.read_csv(path)

        X = extract_dataset_features(df["url"])
        X = X[list(model.feature_names_in_)]

        y_true = df["label"].to_numpy()
        scores = model.predict_proba(X)[:, 1]

        table, ece = calibration_table(y_true, scores)

        brier = brier_score_loss(y_true, scores)
        logloss = log_loss(y_true, scores, labels=[0, 1])

        print(f"\n=== {name} ===")
        print("Records:", len(df))
        print(f"Brier score: {brier:.5f}")
        print(f"Log loss: {logloss:.5f}")
        print(f"ECE (10 bins): {ece:.5f}")

        print("\nCalibration by score interval:")
        print(table.round(4).to_string(index=False))

        filename = name.lower().replace(" ", "_")
        table.to_csv(
            f"results/calibration_{filename}.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
