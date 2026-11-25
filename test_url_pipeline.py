
from src.url_validator import validate_url
from src.url_detector import analyze_url
from src.risk_scoring import calculate_risk
from src.ml_detector import predict_url


TEST_URLS = [
    "https://example.com",
    "https://www.example.org/about",
    "https://login.example.com",
    "http://secure-account-verify.example.com/login",
    "https://account-update.example.net/verify",
    "http://example.org/login",
    "https://example.com/reset-password",
    "https://example.com/very/long/path/with/many/segments",
]


def inspect_url(url):
    print("\n" + "=" * 65)
    print("INPUT:", url)

    validation = validate_url(url)

    if not validation["valid"]:
        print("VALIDATION: FAILED")
        print("REASON:", validation["message"])
        return

    normalized_url = validation["url"]

    try:
        findings = analyze_url(normalized_url)
        risk = calculate_risk(findings)

        if risk is None:
            print("RULE-BASED ANALYSIS: UNAVAILABLE")
            return

        print("HEURISTIC SCORE:", risk["score"])
        print("HEURISTIC LEVEL:", risk["level"])

        print("RULE FINDINGS:")
        if findings:
            for finding in findings:
                print(" -", finding)
        else:
            print(" - None")

        ml_result = predict_url(normalized_url)

        print("ML PREDICTION:", ml_result["prediction"])
        print(
            "ML CLASSIFIER SCORE:",
            f"{ml_result['phishing_score'] * 100:.2f}%"
        )

        rule_flagged = len(findings) > 0
        ml_flagged = ml_result["prediction"] == "Phishing"

        if rule_flagged == ml_flagged:
            print("DETECTOR AGREEMENT: YES")
        else:
            print("DETECTOR AGREEMENT: NO")

    except Exception as error:
        print("TEST ERROR:", type(error).__name__, str(error))


if __name__ == "__main__":
    print("AI CYBERSECURITY GUARDIAN — URL PIPELINE TEST")

    for url in TEST_URLS:
        inspect_url(url)

    print("\nTEST EXECUTION COMPLETED")
