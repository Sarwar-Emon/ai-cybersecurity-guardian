
import re
from urllib.parse import urlsplit


# ==========================================
# 1. SMS PHISHING DETECTION RULES
# ==========================================

SMS_RULES = [
    (
        r"\b(urgent|immediately|act now|final warning)\b",
        "Urgent or high-pressure language detected.",
        10,
    ),
    (
        r"\b(account suspended|account locked|"
        r"account will be suspended|account will be locked)\b",
        "Account suspension or lockout threat detected.",
        20,
    ),
    (
        r"\b(verify your account|verify your identity|"
        r"confirm your information)\b",
        "Account or identity verification request detected.",
        15,
    ),
    (
        r"\b(enter your password|send your password|"
        r"share your otp|provide your otp|"
        r"verification code)\b",
        "Credential or authentication-code request detected.",
        30,
    ),
    (
        r"\b(package delivery failed|delivery failed|"
        r"delivery attempt failed|package held)\b",
        "Delivery-related problem or pressure detected.",
        15,
    ),
    (
        r"\b(pay a fee|unpaid toll|outstanding toll|"
        r"payment(?:\s+is)?\s+required)\b",
        "Payment-related request or threat detected.",
        15,
    ),
    (
        r"\b(click (?:the|this) link|click here|"
        r"tap (?:the|this) link)\b",
        "Link-clicking instruction detected.",
        10,
    ),
]


# ==========================================
# 2. EXTRACT URLS FROM SMS
# ==========================================

def extract_sms_urls(text):
    """
    Extract URLs and bare domains from SMS messages.

    Does not visit links or resolve shortened URLs.
    """

    import tldextract

    pattern = (
        r"(?<![\w@])"
        r"(?:https?://)?"
        r"(?:[a-zA-Z0-9-]+\.)+"
        r"[a-zA-Z]{2,63}"
        r"(?::\d{1,5})?"
        r"(?:/[^\s<>'\"]*)?"
    )

    matches = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    extractor = tldextract.TLDExtract(
        suffix_list_urls=()
    )

    urls = []

    for match in matches:

        candidate = match.rstrip(".,;!?)")

        # Add HTTPS for parsing bare domains.
        # This does NOT verify HTTPS availability.
        if not candidate.lower().startswith(
            ("http://", "https://")
        ):
            normalized = "https://" + candidate
        else:
            normalized = candidate

        try:
            parsed = urlsplit(normalized)
            hostname = parsed.hostname

            if not hostname:
                continue

            extracted = extractor(hostname)

            # Require a recognized public suffix.
            if not extracted.domain or not extracted.suffix:
                continue

            urls.append(normalized)

        except ValueError:
            continue

    return list(dict.fromkeys(urls))


# ==========================================
# 3. ANALYZE SMS
# ==========================================

def analyze_sms(message):
    """
    Analyze an SMS using predefined heuristic rules.

    Scores are not calibrated phishing probabilities.
    """

    text = str(message).strip()

    if not text:
        return {
            "findings": [],
            "score": 0,
            "level": "Not Analyzed",
            "urls": [],
        }

    findings = []
    score = 0

    # Apply phishing detection rules.
    for pattern, description, points in SMS_RULES:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            findings.append(description)
            score += points

    # Extract web links and domains.
    urls = extract_sms_urls(text)

    if urls:
        findings.append(
            f"Found {len(urls)} web link(s) "
            "or domain(s) in the message."
        )
        score += 5

    # Detect explicitly insecure HTTP links.
    # Bare domains are not treated as HTTP.
    if re.search(
        r"(?<![A-Za-z0-9])http://",
        text,
        flags=re.IGNORECASE,
    ):
        findings.append(
            "At least one link uses HTTP instead of HTTPS."
        )
        score += 10

    # Limit score to 100.
    score = min(score, 100)

    # Determine heuristic risk level.
    if score >= 60:
        level = "High Risk"
    elif score >= 30:
        level = "Medium Risk"
    else:
        level = "Low Risk"

    return {
        "findings": list(dict.fromkeys(findings)),
        "score": score,
        "level": level,
        "urls": urls,
    }
