"""
feature_extraction.py
----------------------
Extracts hand-crafted lexical/structural features from a URL string.
These are the same categories of features used in real academic phishing-detection
datasets (e.g., UCI Phishing Websites Dataset), reimplemented from scratch so you
can explain every single feature in an interview.

Feature categories:
1. Length-based       -> phishing URLs tend to be longer / have long hostnames
2. Character-based     -> presence of '@', '-', 'IP address instead of domain'
3. Structural           -> number of subdomains, use of HTTPS, double slashes
4. Keyword-based        -> presence of suspicious trigger words (login, verify, etc.)
5. Shortening services  -> bit.ly, tinyurl, etc. used to hide real destination
"""

import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "bank",
    "confirm", "signin", "webscr", "password", "ebayisapi",
    "suspend", "urgent", "alert", "billing"
]

SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "cutt.ly", "shorte.st"
]

IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)


def _hostname(url: str) -> str:
    parsed = urlparse(url if "://" in url else "http://" + url)
    return parsed.hostname or ""


def extract_features(url: str) -> dict:
    """Convert a raw URL string into a dict of numeric features."""
    url = url.strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.hostname or ""
    path = parsed.path or ""
    full = url.lower()

    features = {
        # 1. Length-based
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(path),

        # 2. Character-based
        "has_at_symbol": int("@" in url),
        "count_dots": url.count("."),
        "count_hyphens": url.count("-"),
        "count_digits": sum(c.isdigit() for c in url),
        "has_ip_address": int(bool(IP_PATTERN.match(host))),
        "has_double_slash_redirect": int("//" in path),

        # 3. Structural
        "uses_https": int(parsed.scheme == "https"),
        "num_subdomains": max(host.count(".") - 1, 0) if host else 0,
        "has_port": int(parsed.port is not None),

        # 4. Keyword-based
        "has_suspicious_word": int(any(w in full for w in SUSPICIOUS_WORDS)),
        "suspicious_word_count": sum(w in full for w in SUSPICIOUS_WORDS),

        # 5. Shortening services
        "is_shortened": int(any(s in full for s in SHORTENING_SERVICES)),
    }
    return features


def extract_features_batch(urls: list) -> "pandas.DataFrame":
    import pandas as pd
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # quick manual sanity check
    samples = [
        "https://www.google.com/search?q=hello",
        "http://192.168.1.5/login/verify-account.php",
        "http://secure-paypal-login.com-update.info/signin",
    ]
    for s in samples:
        print(s)
        print(extract_features(s))
        print()
