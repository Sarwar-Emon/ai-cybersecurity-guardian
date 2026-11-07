
import ipaddress
import re
from urllib.parse import urlsplit


def validate_url(user_input):
    """
    Validate and normalize a URL for analysis.

    Returns:
        {
            "valid": bool,
            "url": str or None,
            "message": str,
            "normalized": bool
        }
    """

    url = str(user_input).strip()

    if not url:
        return {
            "valid": False,
            "url": None,
            "message": "URL cannot be empty.",
            "normalized": False,
        }

    if any(char.isspace() for char in url):
        return {
            "valid": False,
            "url": None,
            "message": "URL cannot contain whitespace.",
            "normalized": False,
        }

    # Reject explicitly unsupported schemes.
    scheme_match = re.match(
        r"^([a-zA-Z][a-zA-Z0-9+.-]*):\/\/",
        url
    )

    if scheme_match:
        scheme = scheme_match.group(1).lower()

        if scheme not in ("http", "https"):
            return {
                "valid": False,
                "url": None,
                "message": "Only HTTP and HTTPS URLs are supported.",
                "normalized": False,
            }

    # Reject malformed scheme prefixes such as http:example.com
    elif re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*:",
        url
    ):
        return {
            "valid": False,
            "url": None,
            "message": "Malformed or unsupported URL scheme.",
            "normalized": False,
        }

    normalized = False

    if not scheme_match:
        if url.startswith("//"):
            return {
                "valid": False,
                "url": None,
                "message": "Please include an explicit URL scheme.",
                "normalized": False,
            }

        url = "https://" + url
        normalized = True

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port

    except ValueError:
        return {
            "valid": False,
            "url": None,
            "message": "Malformed hostname or port.",
            "normalized": normalized,
        }

    if not hostname:
        return {
            "valid": False,
            "url": None,
            "message": "URL must contain a hostname.",
            "normalized": normalized,
        }

    if parsed.username is not None or parsed.password is not None:
        return {
            "valid": False,
            "url": None,
            "message": "URLs containing user credentials are not supported.",
            "normalized": normalized,
        }

    if port is not None and not (1 <= port <= 65535):
        return {
            "valid": False,
            "url": None,
            "message": "Invalid URL port.",
            "normalized": normalized,
        }

    # Validate hostname as an IP address or DNS-style name.
    try:
        ipaddress.ip_address(hostname)

    except ValueError:
        if (
            len(hostname) > 253
            or "." not in hostname
            or hostname.startswith(".")
            or hostname.endswith(".")
        ):
            return {
                "valid": False,
                "url": None,
                "message": "Enter a valid domain name or IP address.",
                "normalized": normalized,
            }

        labels = hostname.split(".")

        for label in labels:
            if not re.fullmatch(
                r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?",
                label
            ):
                return {
                    "valid": False,
                    "url": None,
                    "message": "Invalid hostname label.",
                    "normalized": normalized,
                }

    return {
        "valid": True,
        "url": url,
        "message": (
            "HTTPS was assumed because no scheme was provided."
            if normalized
            else "URL format is valid."
        ),
        "normalized": normalized,
    }
