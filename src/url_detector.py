import ipaddress
import tldextract
from urllib.parse import urlparse


SUSPICIOUS_KEYWORDS = [
    "verify",
    "urgent",
    "login",
    "password",
    "suspended",
    "confirm",
    "update-account"
]

TRUSTED_BRANDS = {
    "paypal": "paypal.com",
    "google": "google.com",
    "microsoft": "microsoft.com",
    "amazon": "amazon.com",
    "apple": "apple.com"
}



def analyze_url(url):
    url = url.strip()
    parsed = urlparse(url)

    findings = []


    # Rule 0: Validate URL structure
    if parsed.scheme not in ("http", "https"):
        return ["Invalid URL: HTTP or HTTPS is required"]

    if not parsed.hostname:
        return ["Invalid URL: Missing hostname"]


    # Rule 1: Check HTTP vs HTTPS
    if parsed.scheme == "http":
        findings.append("URL uses HTTP instead of HTTPS")

    # Rule 2: Check suspicious keywords
    hostname = (parsed.hostname or "").lower()

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in hostname:
            findings.append(
                f"Suspicious keyword in hostname: {keyword}"
            )

    # Rule 3: Check URL length
    if len(url) > 100:
        findings.append("Unusually long URL")

    # Rule 4: Detect IP addresses in the hostname
    if hostname:
        try:
            ipaddress.ip_address(hostname)
            findings.append(
                "URL uses an IP address instead of a domain name"
            )
        except ValueError:
            pass


    # Rule 5: Detect possible brand impersonation
    extracted = tldextract.extract(hostname)


    registered_domain = extracted.top_domain_under_public_suffix

    # Fallback for reserved or unrecognized suffixes
    if not registered_domain:
        if extracted.domain and extracted.suffix:
            registered_domain = (
                f"{extracted.domain}.{extracted.suffix}"
            )
        else:
            registered_domain = extracted.domain

    for brand, official_domain in TRUSTED_BRANDS.items():
        if brand in hostname and registered_domain != official_domain:
            findings.append(
                f"Possible {brand.title()} impersonation: "
                f"actual domain is {registered_domain}"
            )


    return findings
