from functools import lru_cache

from pathlib import Path

import joblib
import pandas as pd

from src.feature_extractor import extract_features


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "diverse"
    / "random_forest_diverse.pkl"
)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_url(url):
    model = load_model()

    features = extract_features(url)

    # Convert feature dictionary into one-row DataFrame.
    X = pd.DataFrame([features])

    # Match the exact feature order used in training.
    expected = list(model.feature_names_in_)

    missing = set(expected) - set(X.columns)

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X = X[expected]

    phishing_score = float(
        model.predict_proba(X)[0, 1]
    )

    predicted_label = int(
        phishing_score >= 0.50
    )

    return {
        "prediction": (
            "Phishing" if predicted_label == 1
            else "Legitimate"
        ),
        "phishing_score": phishing_score,
        "model": "Diverse Random Forest",
        "threshold": 0.50,
    }
