# Phishing URL Detection System

A machine learning system that classifies URLs as **legitimate** or **phishing** using hand-crafted lexical and structural features (no external API calls or blocklists needed — works offline on the URL string alone).

## How it works

1. **Feature extraction** (`src/feature_extraction.py`) — converts a raw URL into 15 numeric features across 5 categories: length-based, character-based, structural, keyword-based, and shortening-service detection.
2. **Dataset** (`data/generate_dataset.py`) — generates a labeled dataset of realistic legitimate and phishing-style URLs, including deliberately hard/overlapping cases (legit login pages, HTTPS phishing sites, label noise) so the model isn't trivially perfect.
3. **Training** (`src/train_model.py`) — trains and compares Logistic Regression vs Random Forest, evaluates with accuracy/precision/recall/F1/confusion matrix, and saves the better model.
4. **Inference** (`src/predict.py`) — CLI tool to classify any new URL with a confidence score.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 94.6% | 95.0% | 94.2% | 94.6% |
| **Random Forest (selected)** | **97.5%** | **97.5%** | **97.5%** | **97.5%** |

See `outputs/confusion_matrix.png` and `outputs/feature_importance.png` for visual breakdowns.

## Setup

```bash
pip install -r requirements.txt
python data/generate_dataset.py     # generates data/urls.csv
python src/train_model.py           # trains model, saves to models/
python src/predict.py "http://192.168.1.5/paypal/login.php"
```

## Project structure

```
phishing-detection-system/
├── data/
│   ├── generate_dataset.py   # synthetic dataset generator
│   └── urls.csv              # generated dataset (url, label)
├── src/
│   ├── feature_extraction.py # URL -> feature vector
│   ├── train_model.py        # training + evaluation pipeline
│   └── predict.py            # CLI inference
├── models/                   # saved model + feature columns (.pkl)
├── outputs/                  # confusion matrix & feature importance plots
├── requirements.txt
└── README.md
```

## Notes on the dataset

This environment has no live internet access to pull a real-world dataset (e.g. UCI Phishing Websites Dataset or a live PhishTank feed), so `generate_dataset.py` programmatically builds a labeled dataset using well-documented, real phishing patterns: IP-address hosts, `@`-symbol redirection tricks, brand-name typosquatting, URL shorteners, and suspicious-keyword subdomains — mixed with legitimate URLs that intentionally include realistic "hard" cases (e.g. real `/account/login` pages) so the model has genuine signal to learn, not just a lookup table.

**To extend this with real-world data:** swap `data/urls.csv` for the UCI Phishing dataset or a PhishTank export — the feature extraction and training pipeline work unchanged on any `url,label` CSV.

## Tech stack

Python, scikit-learn, pandas, NumPy, Matplotlib

## Future improvements

- Add WHOIS-based features (domain age, registrar) — strong phishing signal in practice
- Add live PhishTank/OpenPhish data feed
- Try XGBoost / gradient boosting for comparison
- Package as a Flask API or browser extension for real-time checking
