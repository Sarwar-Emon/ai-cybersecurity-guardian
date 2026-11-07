
import pandas as pd

from src.ml_detector import load_model


model = load_model()

print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")

if not hasattr(model, "feature_importances_"):
    raise TypeError(
        "The loaded model does not expose feature_importances_."
    )

importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_,
})

importance = importance.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

print(importance.to_string(index=False))

print("\nMODEL CLASSES:", model.classes_)

print("\nTOTAL IMPORTANCE:", importance["Importance"].sum())

print("\nFEATURE IMPORTANCE TEST COMPLETED")
