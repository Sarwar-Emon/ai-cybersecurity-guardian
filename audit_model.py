
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.feature_extractor import extract_dataset_features

# Load our trained model and test dataset
model = joblib.load("models/logistic_regression.pkl")
df = pd.read_csv("data/test.csv")

# Extract features
X = extract_dataset_features(df["url"])
y = df["label"]

# Full model predictions
full_pred = model.predict(X)

# Simple length-only baseline
length_pred = (X["url_length"] > 58).astype(int)

print("FULL MODEL")
print("Accuracy:", accuracy_score(y, full_pred))
print("F1:", f1_score(y, full_pred))

print("\nLENGTH-ONLY BASELINE")
print("Accuracy:", accuracy_score(y, length_pred))
print("Precision:", precision_score(
    y, length_pred, zero_division=0
))
print("Recall:", recall_score(y, length_pred))
print("F1:", f1_score(y, length_pred))
print("Confusion Matrix:")
print(confusion_matrix(y, length_pred))

print("\nURL LENGTH BY CLASS")
print(
    df.assign(url_length=X["url_length"])
      .groupby("label")["url_length"]
      .describe()
      .to_string()
)
