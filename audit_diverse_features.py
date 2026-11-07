
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


# ======================================
# 1. LOAD DATA
# ======================================

comp_train = pd.read_csv("data/compphish/train.csv")
comp_val = pd.read_csv("data/compphish/validation.csv")

phi_train = pd.read_csv("data/diverse/phi_train.csv")
phi_eval = pd.read_csv("data/diverse/phi_eval.csv")


# Recreate the same 10,000-row sample
phi_legit = phi_train[
    phi_train["label"] == 0
].sample(n=5000, random_state=42)

phi_phish = phi_train[
    phi_train["label"] == 1
].sample(n=5000, random_state=42)

diverse_train = pd.concat(
    [comp_train, phi_legit, phi_phish],
    ignore_index=True
)


# ======================================
# 2. EXTRACT FEATURES
# ======================================

print("Extracting training features...")
X_train = extract_dataset_features(diverse_train["url"])
y_train = diverse_train["label"]

print("Extracting CompPhish validation features...")
X_comp = extract_dataset_features(comp_val["url"])
y_comp = comp_val["label"]

print("Extracting PhiUSIIL evaluation features...")
X_phi = extract_dataset_features(phi_eval["url"])
y_phi = phi_eval["label"]


# ======================================
# 3. DEFINE ABLATION EXPERIMENTS
# ======================================

experiments = {
    "A - Full": [],
    "B - No HTTPS/path": [
        "https", "path_length"
    ],
    "C - No HTTPS/path/URL length": [
        "https", "path_length", "url_length"
    ],
    "D - Reduced structural": [
        "https",
        "path_length",
        "url_length",
        "hostname_length"
    ]
}


# ======================================
# 4. TRAIN AND EVALUATE
# ======================================

results = []

for experiment, removed in experiments.items():

    features = [
        c for c in X_train.columns
        if c not in removed
    ]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    print(f"\nTraining: {experiment}")
    print("Features:", features)

    model.fit(X_train[features], y_train)

    for dataset_name, X, y in [
        ("CompPhish", X_comp, y_comp),
        ("PhiUSIIL", X_phi, y_phi)
    ]:

        predictions = model.predict(X[features])
        probabilities = model.predict_proba(
            X[features]
        )[:, 1]

        cm = confusion_matrix(
            y, predictions, labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        row = {
            "Experiment": experiment,
            "Dataset": dataset_name,
            "Accuracy": accuracy_score(y, predictions),
            "Precision": precision_score(y, predictions),
            "Recall": recall_score(y, predictions),
            "F1": f1_score(y, predictions),
            "ROC-AUC": roc_auc_score(y, probabilities),
            "False Positives": int(fp),
            "False Negatives": int(fn)
        }

        results.append(row)

        print(f"\n{dataset_name}")
        print(
            f"Accuracy: {row['Accuracy']:.4f} | "
            f"Recall: {row['Recall']:.4f} | "
            f"F1: {row['F1']:.4f} | "
            f"AUC: {row['ROC-AUC']:.4f}"
        )
        print("Confusion matrix:")
        print(cm)


# ======================================
# 5. SAVE RESULTS
# ======================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "models/diverse/ablation_results.csv",
    index=False
)

print("\nFINAL ABLATION COMPARISON")
print(results_df.to_string(index=False))

print("\nAudit complete!")
