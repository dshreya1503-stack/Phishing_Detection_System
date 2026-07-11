"""
train_model.py
---------------
1. Loads data/urls.csv
2. Converts every URL into numeric features (feature_extraction.py)
3. Trains and compares two classifiers: Logistic Regression (baseline, interpretable)
   and Random Forest (stronger, handles non-linear feature interactions)
4. Evaluates with accuracy / precision / recall / F1 / confusion matrix
5. Saves the best model + a feature-importance plot
"""

import os
import sys
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

sys.path.append(os.path.dirname(__file__))
from feature_extraction import extract_features_batch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "urls.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = extract_features_batch(df["url"].tolist())
    y = df["label"]
    return X, y


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix (rows=actual, cols=predicted) [legit, phishing]:")
    print(cm)
    print(classification_report(y_test, preds, target_names=["legitimate", "phishing"]))
    return {"name": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "model": model}


def plot_feature_importance(model, feature_names, path):
    importances = model.feature_importances_
    order = importances.argsort()[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color="#4C72B0")
    plt.title("Random Forest — Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, path, title):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xticks([0, 1], ["legit", "phishing"])
    plt.yticks([0, 1], ["legit", "phishing"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data and extracting features...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_result = evaluate("Logistic Regression", lr, X_test, y_test)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_result = evaluate("Random Forest", rf, X_test, y_test)

    best = max([lr_result, rf_result], key=lambda r: r["f1"])
    print(f"\nBest model by F1-score: {best['name']}")

    joblib.dump(best["model"], os.path.join(MODEL_DIR, "phishing_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.pkl"))
    print(f"Saved best model -> models/phishing_model.pkl")

    if best["name"] == "Random Forest":
        plot_feature_importance(rf, X.columns.tolist(), os.path.join(OUTPUT_DIR, "feature_importance.png"))
        print("Saved feature importance plot -> outputs/feature_importance.png")

    cm = confusion_matrix(y_test, best["model"].predict(X_test))
    plot_confusion_matrix(cm, os.path.join(OUTPUT_DIR, "confusion_matrix.png"), f"{best['name']} — Confusion Matrix")
    print("Saved confusion matrix plot -> outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
