
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from src.feature_extractor import extract_dataset_features


# Reconstruct the original diverse training data
comp = pd.read_csv("data/compphish/train.csv")
phi = pd.read_csv("data/diverse/phi_train.csv")

phi_0 = phi[phi["label"] == 0].sample(
    n=5000, random_state=42
)
phi_1 = phi[phi["label"] == 1].sample(
    n=5000, random_state=42
)

train = pd.concat(
    [comp, phi_0, phi_1],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

X_train = extract_dataset_features(train["url"])
y_train = train["label"]

# Load the existing deployed model
original = joblib.load(
    "models/diverse/random_forest_diverse.pkl"
)

# Train the candidate without HTTPS
columns = [c for c in X_train.columns if c != "https"]

candidate = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

candidate.fit(X_train[columns], y_train)

# Evaluate on the same CompPhish test data
test = pd.read_csv("data/compphish/test.csv")

X_test = extract_dataset_features(test["url"])
y_test = test["label"]

models = {
    "Original Random Forest": (
        original,
        list(original.feature_names_in_)
    ),
    "Without HTTPS": (
        candidate,
        columns
    )
}

for name, (model, features) in models.items():

    predictions = model.predict(X_test[features])

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    print(f"\n=== {name} ===")
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

    print("False Positive Rate:", f"{fp / (fp + tn) * 100:.2f}%")
    print("False Negative Rate:", f"{fn / (fn + tp) * 100:.2f}%")

print("\nERROR COMPARISON COMPLETED")
