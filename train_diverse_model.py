
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
    confusion_matrix,
)

from src.feature_extractor import extract_dataset_features


# ==========================================
# 1. LOAD DATA
# ==========================================

comp_train = pd.read_csv("data/compphish/train.csv")
comp_val = pd.read_csv("data/compphish/validation.csv")

phi_train = pd.read_csv("data/diverse/phi_train.csv")
phi_eval = pd.read_csv("data/diverse/phi_eval.csv")


# ==========================================
# 2. SAMPLE PHIUSIIL TRAINING DATA
# ==========================================

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

# Combine with CompPhish training data
diverse_train = pd.concat(
    [comp_train, phi_sample],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

print("Original training URLs:", len(comp_train))
print("Additional PhiUSIIL URLs:", len(phi_sample))
print("Diverse training URLs:", len(diverse_train))

print("\nDiverse training class distribution:")
print(diverse_train["label"].value_counts())


# ==========================================
# 3. EXTRACT FEATURES
# ==========================================

print("\nExtracting training features...")

X_comp_train = extract_dataset_features(
    comp_train["url"]
)

X_diverse_train = extract_dataset_features(
    diverse_train["url"]
)

print("Extracting CompPhish validation features...")
X_comp_val = extract_dataset_features(
    comp_val["url"]
)

print("Extracting PhiUSIIL evaluation features...")
X_phi_eval = extract_dataset_features(
    phi_eval["url"]
)


# ==========================================
# 4. TRAIN MODELS
# ==========================================

def make_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )


baseline = make_model()
diverse = make_model()

print("\nTraining baseline Random Forest...")
baseline.fit(
    X_comp_train,
    comp_train["label"]
)

print("Training diverse Random Forest...")
diverse.fit(
    X_diverse_train,
    diverse_train["label"]
)


# ==========================================
# 5. EVALUATE MODELS
# ==========================================

results = []


def evaluate(model, model_name, dataset_name, X, y):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1": f1_score(
            y, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y, probabilities),
    }

    results.append(metrics)

    print(f"\n{model_name} | {dataset_name}")

    for key in [
        "Accuracy", "Precision",
        "Recall", "F1", "ROC-AUC"
    ]:
        print(f"{key}: {metrics[key]:.4f}")

    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))


for model_name, model in [
    ("Baseline", baseline),
    ("Diverse", diverse)
]:

    evaluate(
        model,
        model_name,
        "CompPhish Validation",
        X_comp_val,
        comp_val["label"]
    )

    evaluate(
        model,
        model_name,
        "PhiUSIIL Evaluation",
        X_phi_eval,
        phi_eval["label"]
    )


# ==========================================
# 6. SAVE RESULTS
# ==========================================

output = Path("models/diverse")
output.mkdir(parents=True, exist_ok=True)

joblib.dump(
    diverse,
    output / "random_forest_diverse.pkl"
)

results_df = pd.DataFrame(results)

results_df.to_csv(
    output / "comparison_results.csv",
    index=False
)

print("\nFINAL COMPARISON")
print(results_df.to_string(index=False))

print("\nResults saved successfully!")
