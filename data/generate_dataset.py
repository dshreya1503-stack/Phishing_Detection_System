"""
generate_dataset.py
--------------------
Generates a labeled dataset of legitimate vs. phishing-style URLs.

NOTE: In a real deployment you'd use a public dataset such as the UCI Phishing
Websites Dataset or PhishTank's live feed. This generator programmatically
creates realistic examples of both classes using well-documented phishing
patterns (IP-based hosts, '@' redirection tricks, brand-name + suspicious
subdomain combos, URL shorteners, etc.) so the project is fully runnable and
reproducible offline.
"""

import random
import csv
import os

random.seed(42)

LEGIT_DOMAINS = [
    "google.com", "amazon.com", "wikipedia.org", "github.com", "microsoft.com",
    "apple.com", "netflix.com", "linkedin.com", "stackoverflow.com", "nytimes.com",
    "bbc.com", "reddit.com", "spotify.com", "dropbox.com", "adobe.com",
    "uttaranchaluniversity.ac.in", "irctc.co.in", "icicibank.com", "hdfcbank.com",
]

LEGIT_PATHS = [
    "", "/search?q=weather", "/products/laptop", "/en/wiki/Artificial_intelligence",
    "/user/settings", "/blog/2026/07/release-notes", "/docs/api/v2",
    "/careers/software-engineer", "/account/orders", "/watch?v=abc123",
]

PHISH_BRAND_KEYWORDS = ["paypal", "amazon", "netflix", "icicibank", "hdfcbank", "apple", "microsoft"]
PHISH_TRIGGER_WORDS = ["login", "verify", "update", "secure", "account", "confirm", "signin", "suspend"]
SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "cutt.ly"]

TLD_JUNK = [".info", ".xyz", ".top", ".click", ".biz", ".ru", ".tk"]


def random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


LEGIT_ACCOUNT_PATHS = [
    "/account/login", "/account/verify-email", "/user/update-profile",
    "/secure/signin", "/account/confirm-signup", "/support/login-help",
]


def make_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    # ~25% of the time, use a legit path that itself contains a "suspicious"
    # word like login/verify/secure -- this is realistic (real sites have
    # login pages!) and forces the model to rely on more than keyword-matching.
    if random.random() < 0.25:
        path = random.choice(LEGIT_ACCOUNT_PATHS)
    else:
        path = random.choice(LEGIT_PATHS)
    scheme = "https" if random.random() < 0.92 else "http"  # small amount of legit http too
    return f"{scheme}://www.{domain}{path}"


def make_phishing_url():
    style = random.choice(["ip", "at_symbol", "brand_subdomain", "shortener", "typo_domain", "clean_looking"])
    brand = random.choice(PHISH_BRAND_KEYWORDS)
    trigger = random.choice(PHISH_TRIGGER_WORDS)

    if style == "ip":
        return f"http://{random_ip()}/{brand}/{trigger}.php"

    if style == "at_symbol":
        return f"http://{brand}.com@malicious-server{random.randint(1,999)}.com/{trigger}"

    if style == "brand_subdomain":
        junk_tld = random.choice(TLD_JUNK)
        return f"http://{brand}-{trigger}-secure.{random.choice(['account','support','portal'])}{junk_tld}/{trigger}"

    if style == "shortener":
        return f"http://{random.choice(SHORTENERS)}/{brand}{random.randint(100,999)}"

    if style == "typo_domain":
        # subtle brand misspelling / extra hyphen, classic typosquatting
        typo = brand.replace("a", "4", 1) if "a" in brand else brand + "1"
        return f"http://www.{typo}-{trigger}.com/{trigger}.html"

    if style == "clean_looking":
        # a harder case: no IP, no @, no shortener, no obvious trigger word --
        # just a plausible-looking junk-TLD domain, sometimes even over HTTPS
        # (real phishing sites increasingly use free HTTPS certs). Forces the
        # model to weigh weaker/combined signals instead of one obvious flag.
        junk_tld = random.choice(TLD_JUNK)
        fake_word = random.choice(["portal", "hub", "center", "online", "services"])
        scheme = "https" if random.random() < 0.35 else "http"
        return f"{scheme}://www.{brand}{fake_word}{junk_tld}/home"


def generate(n_per_class=500, label_noise=0.03):
    rows = []
    for _ in range(n_per_class):
        rows.append((make_legit_url(), 0))       # 0 = legitimate
    for _ in range(n_per_class):
        rows.append((make_phishing_url(), 1))    # 1 = phishing

    # Inject a small amount of label noise (~3%) to simulate real-world
    # annotation ambiguity/imperfect ground truth -- avoids an unrealistic
    # 100% test accuracy and gives the model genuine edge cases to fail on.
    n_noisy = int(len(rows) * label_noise)
    noisy_idxs = random.sample(range(len(rows)), n_noisy)
    for i in noisy_idxs:
        url, label = rows[i]
        rows[i] = (url, 1 - label)

    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "urls.csv")
    rows = generate(n_per_class=600)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(rows)
    print(f"Generated {len(rows)} rows -> {out_path}")
