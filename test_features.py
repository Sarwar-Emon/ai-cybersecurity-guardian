
from src.feature_extractor import extract_features

url = "http://paypal-login-verify.example.com"

features = extract_features(url)

for name, value in features.items():
    print(f"{name}: {value}")
