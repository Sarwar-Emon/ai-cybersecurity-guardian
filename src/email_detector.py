
import re
from urllib.parse import urlsplit

from src.sender_analyzer import analyze_sender


# ==========================================
# 1. EMAIL CONTENT RULES
# ==========================================

EMAIL_RULES = [
    (
        r"\b(urgent|immediately|act now|right away)\b",
        "Urgent or high-pressure language detected.",
        10,
    ),
    (
        r"\b(account suspended|account will be suspended|"
        r"account locked|account will be locked|"
        r"account terminated)\b",
        "Account suspension or lockout threat detected.",
        20,
    ),
    (
        r"\b(verify your account|verify your identity|"
        r"confirm your account|update your account)\b",
        "Account verification request detected.",
        15,
    ),
    (
        r"\b(send your password|enter your password|"
        r"provide your password|share your password|"
        r"provide your otp|share your otp)\b",
        "Request for credentials or authentication codes detected.",
        30,
    ),
    (
        r"\b(click here|click the link|open this link)\b",
        "Link-clicking instruction detected.",
        10,
    ),
]


# ==========================================
# 2. EXTRACT EMAIL URLS
# ==========================================

def extract_email_urls(text):
    """
    Extract HTTP/HTTPS URLs without visiting them.
    """

    pattern = r"https?://[^\s<>'\"]+"

    matches = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE
    )

    urls = []

    for match in matches:
        url = match.rstrip(".,;!?)")

        try:
            parsed = urlsplit(url)

            if (
                parsed.scheme.lower() in ("http", "https")
                and parsed.hostname
            ):
                urls.append(url)

        except ValueError:
            continue

    return list(dict.fromkeys(urls))


# ==========================================
# 3. EMAIL ANALYSIS
# ==========================================

def analyze_email(email_text):
    """
    Combine email content rules with sender analysis.

    Scores are heuristic, not calibrated probabilities.
    """

    text = str(email_text).strip()

    if not text:
        return {
            "findings": [],
            "score": 0,
            "level": "Not Analyzed",
            "urls": [],
            "sender_analysis": None,
            "content_score": 0,
            "sender_score": 0,
        }

    findings = []
    content_score = 0

    # --------------------------------------
    # A. CONTENT ANALYSIS
    # --------------------------------------

    for pattern, description, points in EMAIL_RULES:

        if re.search(pattern, text, flags=re.IGNORECASE):
            findings.append(description)
            content_score += points

    # --------------------------------------
    # B. URL EXTRACTION
    # --------------------------------------

    urls = extract_email_urls(text)

    if urls:
        findings.append(
            f"Found {len(urls)} HTTP/HTTPS URL(s) in the email."
        )
        content_score += 5

    if any(
        url.lower().startswith("http://")
        for url in urls
    ):
        findings.append(
            "At least one link uses HTTP instead of HTTPS."
        )
        content_score += 10

    # --------------------------------------
    # C. SENDER IMPERSONATION ANALYSIS
    # --------------------------------------

    sender_result = analyze_sender(text)

    sender_findings = sender_result["findings"]
    sender_score = sender_result["score"]

    findings.extend(sender_findings)

    # --------------------------------------
    # D. COMBINED HEURISTIC SCORE
    # --------------------------------------

    score = min(content_score + sender_score, 100)

    if score >= 60:
        level = "High Risk"

    elif score >= 30:
        level = "Medium Risk"

    else:
        level = "Low Risk"

    # --------------------------------------
    # E. RETURN RESULTS
    # --------------------------------------

    return {
        "findings": list(dict.fromkeys(findings)),
        "score": score,
        "level": level,
        "urls": urls,
        "sender_analysis": sender_result,
        "content_score": min(content_score, 100),
        "sender_score": sender_score,
    }
