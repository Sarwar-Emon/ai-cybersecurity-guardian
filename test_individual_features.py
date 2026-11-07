from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.feature_extractor import extract_dataset_features
from src.feature_extractor_v2 import extract_features_v2


ADDITIONAL_FEATURES = [
    "hostname_digit_count",
    "actual_subdomain_count",
    "suspicious_file_extension",
]

EXPERIMENTS = {
    "Baseline 11": [],
    "+ Hostname digits": ["hostname_digit_count"],
    "+ Actual subdomains": ["actual_subdomain_count"],
    "+ Suspicious extension": ["suspicious_file_extension"],
    "All three (14)": ADDITIONAL_FEATURES,
}


def load_data():
    comp_train = pd.read_csv("data/compphish/train.csv")
    comp_val = pd.read_csv("data/compphish/validation.csv")
    phi_train = pd.read_csv("data/diverse/phi_train.csv")
    phi_eval = pd.read_csv("data/diverse/phi_eval.csv")

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
        frac=1, random_state=42
    ).reset_index(drop=True)

    return train, comp_val, phi_eval


def get_feature_table(urls):
    # Build all 14 features once per dataset.
    return pd.DataFrame(
        [extract_features_v2(url) for url in urls]
    )


def evaluate(model, X, y, experiment, dataset):
    predictions = model.predict(X)
    scores = model.predict_proba(X)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Experiment": experiment,
        "Dataset": dataset,
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

    print("Extracting features...")
    X_train_all = get_feature_table(train["url"])
    X_comp_all = get_feature_table(comp_val["url"])
    X_phi_all = get_feature_table(phi_eval["url"])

    original_columns = list(
        extract_dataset_features(
            train["url"].iloc[:1]
        ).columns
    )

    assert len(original_columns) == 11
    assert len(X_train_all.columns) == 14

    results = []

    for experiment, added in EXPERIMENTS.items():
        columns = original_columns + added

        print(f"\nTraining: {experiment}")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train_all[columns],
            train["label"]
        )

        for name, X_all, df in [
            ("CompPhish Validation", X_comp_all, comp_val),
            ("PhiUSIIL Evaluation", X_phi_all, phi_eval),
        ]:
            result = evaluate(
                model,
                X_all[columns],
                df["label"],
                experiment,
                name,
            )

            results.append(result)

            print(
                f"{name}: "
                f"F1={result['F1']:.4f}, "
                f"Recall={result['Recall']:.4f}, "
                f"FP={result['FP']}, "
                f"FN={result['FN']}"
            )

    results_df = pd.DataFrame(results)

    print("\n=== INDIVIDUAL FEATURE COMPARISON ===")
    print(results_df.to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    results_df.to_csv(
        "results/individual_feature_ablation.csv",
        index=False
    )

    print("\nSaved: results/individual_feature_ablation.csv")


if __name__ == "__main__":
    main()
