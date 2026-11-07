
from src.url_detector import analyze_url

test_urls = [
    "https://www.google.com",
    "http://paypal-login-verify.example.com",
    "https://example.com"
]

for url in test_urls:
    print("\nURL:", url)
    print("Findings:", analyze_url(url))
