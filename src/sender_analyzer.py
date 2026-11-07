
import re
from email.utils import parseaddr

import tldextract


# ==========================================
# 1. TRUSTED BRAND DOMAINS
# ==========================================

# Demonstration allowlist, not a complete list
# of official domains.
TRUSTED_DOMAINS = {
    "paypal": "paypal.com",
    "google": "google.com",
    "microsoft": "microsoft.com",
    "amazon": "amazon.com",
    "apple": "apple.com",
}
SECURITY_KEYWORDS = {
    "support",
    "security",
    "verify",
    "verification",
    "account",
    "login",
    "update",
    "billing",
    "secure",
}

LOOKALIKE_MAP = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "5": "s",
    "7": "t",
})

# Use packaged Public Suffix List data.
EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=()
)


# ==========================================
# 2. REGISTERED DOMAIN EXTRACTION
# ==========================================

def get_registered_domain(domain):
    """Extract the registrable domain."""

    extracted = EXTRACTOR(domain)

    if not extracted.domain or not extracted.suffix:
        return ""

    return f"{extracted.domain}.{extracted.suffix}"


# ==========================================
# 3. EXTRACT EMAIL SENDER
# ==========================================

def extract_sender(email_text):
    """Extract sender from the first From header."""

    for line in str(email_text).splitlines():

        match = re.match(
            r"^From:\s*(.+)$",
            line,
            re.IGNORECASE,
        )

        if match:
            display_name, address = parseaddr(
                match.group(1)
            )

            return (
                display_name.strip(),
                address.lower().strip(),
            )

    return "", ""


# ==========================================
# 4. TRUSTED DOMAIN CHECK
# ==========================================

def is_trusted_domain(domain, trusted_domain):
    """Check exact domain or genuine subdomain."""

    return (
        domain == trusted_domain
        or domain.endswith("." + trusted_domain)
    )


# ==========================================
# 5. LEVENSHTEIN LOOKALIKE DETECTION
# ==========================================

def is_close_lookalike(candidate, brand):
    """
    Detect a one-character difference.

    Examples:
        paypai -> paypal : True
        paypa1 -> paypal : True
        paypal -> paypal : False
    """

    candidate = candidate.lower()
    brand = brand.lower()

    if candidate == brand:
        return False

    if abs(len(candidate) - len(brand)) > 1:
        return False

    previous = list(range(len(brand) + 1))

    for i, char_a in enumerate(candidate, start=1):

        current = [i]

        for j, char_b in enumerate(brand, start=1):

            insertion = current[j - 1] + 1
            deletion = previous[j] + 1

            substitution = (
                previous[j - 1]
                + (char_a != char_b)
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution,
                )
            )

        previous = current

    return previous[-1] == 1


# ==========================================
# 6. MAIN SENDER ANALYSIS
# ==========================================

def analyze_sender(email_text):
    """
    Detect basic email sender impersonation.

    Uses:
        - Registered-domain parsing
        - Character substitution detection
        - Levenshtein distance
        - Misleading hostname detection
        - Display-name mismatch detection

    Does not authenticate sender identity.
    """

    display_name, address = extract_sender(
        email_text
    )

    findings = []
    score = 0

    result = {
        "sender": address,
        "domain": "",
        "registered_domain": "",
        "findings": findings,
        "score": 0,
    }

    if not address:
        return result

    # --------------------------------------
    # A. BASIC ADDRESS VALIDATION
    # --------------------------------------

    if address.count("@") != 1:

        findings.append(
            "Sender address has an invalid structure."
        )

        result["score"] = 15
        return result

    domain = address.rsplit("@", 1)[1].rstrip(".")

    result["domain"] = domain

    labels = domain.split(".")

    if (
        not domain
        or any(
            not label
            or len(label) > 63
            or not re.fullmatch(
                r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
                label,
            )
            for label in labels
        )
    ):

        findings.append(
            "Sender domain has an invalid structure."
        )

        result["score"] = 15
        return result

    # --------------------------------------
    # B. REGISTERED DOMAIN
    # --------------------------------------

    registered_domain = get_registered_domain(
        domain
    )

    result["registered_domain"] = registered_domain

    if not registered_domain:

        findings.append(
            "Sender domain could not be reliably classified."
        )

        result["score"] = 10
        return result

    extracted = EXTRACTOR(domain)

    domain_label = extracted.domain.lower()

    normalized_label = domain_label.translate(
        LOOKALIKE_MAP
    )

    display_lower = display_name.lower()

    # --------------------------------------
    # C. BRAND IMPERSONATION RULES
    # --------------------------------------

    for brand, trusted_domain in TRUSTED_DOMAINS.items():

        # Skip domains belonging to the trusted
        # domain namespace.
        # This is not proof of email authentication.
        if is_trusted_domain(domain, trusted_domain):
            continue

        # RULE 1:
        # Lookalike registered-domain label
        #
        # paypa1.com -> paypal.com
        # paypai.com -> paypal.com

        if (
            (
                normalized_label == brand
                or is_close_lookalike(
                    domain_label,
                    brand,
                )
            )
            and registered_domain != trusted_domain
        ):

            findings.append(
                f"Sender domain resembles {trusted_domain} "
                "but does not match it."
            )

            score += 30

        # RULE: Brand + security-related keyword
        #
        # Examples:
        # paypal-support.com
        # google-security.com
        # amazon-verification.com
        #
        # Inspect only the registered-domain label,
        # not arbitrary subdomains.

        domain_parts = domain_label.split("-")

        brand_keyword_match = (
            brand in domain_parts
            and any(
                keyword in domain_parts
                for keyword in SECURITY_KEYWORDS
            )
        )

        if brand_keyword_match:

            findings.append(
                f"Sender domain combines the brand "
                f"'{brand}' with a security-related "
                "keyword in its registered domain."
            )

            score += 15


        # RULE 2:
        # Trusted domain embedded in another
        # hostname.
        #
        # paypal.com.security.example.com

        if trusted_domain + "." in domain:

            findings.append(
                f"Sender domain embeds {trusted_domain} "
                "inside a different hostname."
            )

            score += 30

        # RULE 3:
        # Display-name mismatch.
        #
        # PayPal Support <support@random-example.com>

        if (
            re.search(
                rf"\b{re.escape(brand)}\b",
                display_lower,
            )
            and not is_trusted_domain(
                domain,
                trusted_domain,
            )
        ):

            findings.append(
                f"Display name mentions {brand}, "
                f"but sender domain is not {trusted_domain}."
            )

            score += 10

    # --------------------------------------
    # D. FINAL RESULT
    # --------------------------------------

    result["findings"] = list(
        dict.fromkeys(findings)
    )

    result["score"] = min(score, 100)

    return result
