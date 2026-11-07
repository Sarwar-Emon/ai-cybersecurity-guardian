import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from src.feature_extractor import extract_dataset_features
from src.feature_extractor_v2 import extract_features_v2


def load_data():
    comp_train = pd.read_csv("data/compphish/train.csv")
    comp_val = pd.read_csv("data/compphish/validation.csv")
    phi_train = pd.read_csv("data/diverse/phi_train.csv")
    phi_eval = pd.read_csv("data/diverse/phi_eval.csv")

    # Reproduce the original diverse-training sample
    phi_legitimate = phi_train[
        phi_train["label"] == 0
    ].sample(n=5000, random_state=42)

    phi_phishing = phi_train[
        phi_train["label"] == 1
    ].sample(n=5000, random_state=42)

    phi_sample = pd.concat(
        [phi_legitimate, phi_phishing],
        ignore_index=True
    )

    train = pd.concat(
    [comp_train, phi_sample],
    ignore_index=True
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

    return train, comp_val, phi_eval


def get_features(urls, version):
    if version == 11:
        return extract_dataset_features(urls)
    return pd.DataFrame([
        extract_features_v2(url) for url in urls
    ])


def evaluate(model, X, y, dataset_name, version):
    predictions = model.predict(X)
    scores = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Features": version,
        "Dataset": dataset_name,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions),
        "Recall": recall_score(y, predictions),
        "F1": f1_score(y, predictions),
        "ROC-AUC": roc_auc_score(y, scores),
        "FP": int(fp),
        "FN": int(fn),
    }


def main():
    train, comp_val, phi_eval = load_data()

    print("Training records:", len(train))

    results = []

    for version in [11, 14]:
        print(f"\nTraining {version}-feature model...")

        X_train = get_features(train["url"], version)
        y_train = train["label"].to_numpy()

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        for name, df in [
            ("CompPhish Validation", comp_val),
            ("PhiUSIIL Evaluation", phi_eval),
        ]:
            X = get_features(df["url"], version)
            y = df["label"].to_numpy()

            result = evaluate(
                model, X, y, name, version
            )
            results.append(result)

            print(
                f"{name}: "
                f"Recall={result['Recall']:.4f}, "
                f"F1={result['F1']:.4f}, "
                f"FP={result['FP']}, "
                f"FN={result['FN']}"
            )

    results_df = pd.DataFrame(results)
    print("\n=== FULL COMPARISON ===")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        "results/feature_ablation_v2.csv",
        index=False
    )


if __name__ == "__main__":
    main()
