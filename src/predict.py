"""
predict.py
----------
Load the trained model and classify one or more URLs.

Usage:
    python src/predict.py "http://192.168.1.5/paypal/login.php"
    python src/predict.py "https://www.google.com" "http://bit.ly/free-prize99"
"""

import sys
import os
import joblib
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from feature_extraction import extract_features

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "phishing_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")


def load_model():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns


def predict_url(url, model, columns):
    feats = extract_features(url)
    X = pd.DataFrame([feats])[columns]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][pred]
    label = "PHISHING" if pred == 1 else "legitimate"
    return label, proba


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <url> [<url2> ...]")
        sys.exit(1)

    model, columns = load_model()
    for url in sys.argv[1:]:
        label, confidence = predict_url(url, model, columns)
        print(f"{url}\n  -> {label}  (confidence: {confidence:.2%})\n")
