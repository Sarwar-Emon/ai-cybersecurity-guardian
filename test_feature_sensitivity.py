
from src.feature_extractor import extract_features
from src.ml_detector import predict_url

BASE_URL = "https://example.com"

TEST_URLS = [
    BASE_URL,
    "https://example.com/a",
    "https://example.com/abcdefghij",
    "https://example.com/abcdefghijklmnopqrst",
    "https://example.com/a-b",
    "https://example.com/abc",
    "http://example.com",
    "https://example.com/a/b/c/d",
]

baseline = extract_features(BASE_URL)

print("\n=== RANDOM FOREST FEATURE SENSITIVITY ===")

for url in TEST_URLS:
    features = extract_features(url)
    result = predict_url(url)

    changed = {
        key: (baseline[key], value)
        for key, value in features.items()
        if value != baseline[key]
    }

    print("\nURL:", url)
    print("Changed features:", changed)
    print("ML prediction:", result["prediction"])
    print(
        "ML phishing-class score:",
        f"{result['phishing_score'] * 100:.2f}%"
    )

print("\nEXPERIMENT COMPLETED")
