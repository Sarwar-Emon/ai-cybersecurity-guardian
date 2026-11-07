
def calculate_risk(findings):
    score = 0

    for finding in findings:
        if finding.startswith("Invalid URL:"):
            return None

        if "HTTP instead of HTTPS" in finding:
            score += 10

        elif "Suspicious keyword" in finding:
            score += 10

        elif "Unusually long URL" in finding:
            score += 5

        elif "IP address" in finding:
            score += 20

        elif "impersonation" in finding.lower():
            score += 35

    score = min(score, 100)

    if score >= 60:
        level = "High Risk"
    elif score >= 30:
        level = "Medium Risk"
    else:
        level = "Low Risk"

    return {
        "score": score,
        "level": level
    }
