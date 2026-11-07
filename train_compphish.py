
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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


# =====================================
# 1. LOAD TRAINING AND VALIDATION DATA
# =====================================

train = pd.read_csv("data/compphish/train.csv")
val = pd.read_csv("data/compphish/validation.csv")

print("Training URLs:", len(train))
print("Validation URLs:", len(val))


# =====================================
# 2. EXTRACT URL FEATURES
# =====================================

print("\nExtracting training features...")
X_train = extract_dataset_features(train["url"])
y_train = train["label"]

print("Extracting validation features...")
X_val = extract_dataset_features(val["url"])
y_val = val["label"]

assert X_train.columns.tolist() == X_val.columns.tolist()
assert not X_train.isnull().any().any()
assert not X_val.isnull().any().any()


# =====================================
# 3. DEFINE ML MODELS
# =====================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}


# =====================================
# 4. TRAIN AND EVALUATE
# =====================================

results = []

Path("models/compphish").mkdir(
    parents=True,
    exist_ok=True
)

for name, model in models.items():

    print(f"\n{'=' * 45}")
    print(f"TRAINING: {name}")
    print("=" * 45)

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    auc = roc_auc_score(y_val, probabilities)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, predictions))

    filename = (
        "logistic_regression.pkl"
        if name == "Logistic Regression"
        else "random_forest.pkl"
    )

    joblib.dump(
        model,
        f"models/compphish/{filename}"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": auc
    })


# =====================================
# 5. SAVE VALIDATION RESULTS
# =====================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "models/compphish/validation_results.csv",
    index=False
)

print("\nFINAL VALIDATION COMPARISON")
print(results_df.to_string(index=False))

print("\nModels and results saved successfully!")
