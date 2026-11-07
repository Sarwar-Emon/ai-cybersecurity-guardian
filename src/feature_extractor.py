
import ipaddress
from urllib.parse import urlparse

import pandas as pd


def extract_features(url):
    """Extract numerical features from a URL string."""

    url = str(url).strip()
    parsed = urlparse(url)

    hostname = (parsed.hostname or "").lower()

    # Check whether hostname is an IP address
    has_ip = 0

    if hostname:
        try:
            ipaddress.ip_address(hostname)
            has_ip = 1
        except ValueError:
            pass

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "digit_count": sum(c.isdigit() for c in url),
        "special_char_count": sum(
            c in "@?=&%_" for c in url
        ),
        "https": int(parsed.scheme == "https"),
        "has_ip": has_ip,
        "subdomain_dots": hostname.count("."),
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
    }

    return features


def extract_dataset_features(urls):
    """Convert multiple URLs into a feature DataFrame."""

    return pd.DataFrame(
        [extract_features(url) for url in urls]
    )
