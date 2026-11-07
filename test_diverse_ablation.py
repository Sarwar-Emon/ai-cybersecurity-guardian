
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.feature_extractor import extract_dataset_features


# 1. Load the same training sources as the deployed model
comp_train = pd.read_csv("data/compphish/train.csv")
phi_train = pd.read_csv("data/diverse/phi_train.csv")

phi_legitimate = phi_train[phi_train["label"] == 0].sample(
    n=5000, random_state=42
)
phi_phishing = phi_train[phi_train["label"] == 1].sample(
    n=5000, random_state=42
)

train = pd.concat(
    [comp_train, phi_legitimate, phi_phishing],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)


# 2. Load held-out evaluation datasets
eval_sets = {
    "CompPhish Test": pd.read_csv(
        "data/compphish/test.csv"
    ),
    "PhiUSIIL Evaluation": pd.read_csv(
        "data/diverse/phi_eval.csv"
    ),
}


# 3. Extract features
print("Extracting training features...")

X_train = extract_dataset_features(train["url"])
y_train = train["label"]

eval_features = {}

for name, df in eval_sets.items():
    print("Extracting:", name)
    eval_features[name] = (
        extract_dataset_features(df["url"]),
        df["label"]
    )


# 4. Define controlled experiments
experiments = {
    "All features": list(X_train.columns),

    "Without HTTPS": [
        c for c in X_train.columns
        if c != "https"
    ],

    "Without HTTPS and path length": [
        c for c in X_train.columns
        if c not in ["https", "path_length"]
    ],

    "Without HTTPS, path length, URL length": [
        c for c in X_train.columns
        if c not in [
            "https", "path_length", "url_length"
        ]
    ],
}


# 5. Train and evaluate
results = []

for experiment, columns in experiments.items():

    print(f"\nTraining: {experiment}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train[columns], y_train)

    for dataset_name, (X_eval, y_eval) in eval_features.items():

        X = X_eval[columns]

        predictions = model.predict(X)
        scores = model.predict_proba(X)[:, 1]

        row = {
            "Experiment": experiment,
            "Dataset": dataset_name,
            "Accuracy": accuracy_score(y_eval, predictions),
            "Precision": precision_score(
                y_eval, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_eval, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_eval, predictions, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_eval, scores),
        }

        results.append(row)


# 6. Display results
results_df = pd.DataFrame(results)

print("\n=== DIVERSE MODEL ABLATION RESULTS ===")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nEXPERIMENT COMPLETED")
