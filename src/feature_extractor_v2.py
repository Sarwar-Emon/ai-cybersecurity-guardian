from pathlib import PurePosixPath
from urllib.parse import urlparse

import tldextract

from src.feature_extractor import extract_features


SUSPICIOUS_EXTENSIONS = {
    ".exe", ".scr", ".bat", ".cmd", ".ps1",
    ".vbs", ".js", ".jar", ".apk", ".msi",
}


def extract_features_v2(url):
    """Original 11 features plus 3 experimental features."""

    url = str(url).strip()
    features = extract_features(url)

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    # Count digits in hostname only
    hostname_digit_count = sum(
        char.isdigit() for char in hostname
    )

    # Count subdomain levels using the public suffix list
    domain_parts = tldextract.extract(hostname)
    subdomain = domain_parts.subdomain

    actual_subdomain_count = (
        len(subdomain.split(".")) if subdomain else 0
    )

    # Check the extension of the URL path, not its query
    extension = PurePosixPath(
        parsed.path.lower()
    ).suffix

    suspicious_file_extension = int(
        extension in SUSPICIOUS_EXTENSIONS
    )

    features.update({
        "hostname_digit_count": hostname_digit_count,
        "actual_subdomain_count": actual_subdomain_count,
        "suspicious_file_extension": suspicious_file_extension,
    })

    return features
