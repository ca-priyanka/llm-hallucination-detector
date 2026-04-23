"""
Training pipeline for hallucination classifier.

Usage:
    python train.py

Outputs:
    models/hallucination_model.joblib   – trained pipeline
    models/feature_names.json           – feature order
    reports/metrics.json                – evaluation metrics
    reports/pr_curve.png                – precision-recall chart
"""

import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from data.dataset import SAMPLES
from features.extractor import extract_features, features_to_array

FEATURE_NAMES = [
    "semantic_sim",
    "confidence_score",
    "ner_mismatch",
    "contradiction_score",
    "numeric_mismatch",
]

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# ── Build feature matrix ──────────────────────────────────────────────────────

def build_feature_matrix(samples: list[dict], cache_path="models/features_cache.json") -> tuple:
    """Extract features for all samples (with disk cache to avoid re-computing)."""

    if os.path.exists(cache_path):
        print(f"  Loading cached features from {cache_path}")
        with open(cache_path) as f:
            cache = json.load(f)
        X = np.array(cache["X"], dtype=np.float32)
        y = np.array(cache["y"], dtype=int)
        return X, y

    print(f"  Extracting features for {len(samples)} samples (this may take a few minutes)...")
    rows = []
    labels = []

    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {s['question'][:50]}...")
        feat = extract_features(s["question"], s["answer"])
        rows.append(features_to_array(feat))
        labels.append(s["label"])

    X = np.stack(rows)
    y = np.array(labels, dtype=int)

    with open(cache_path, "w") as f:
        json.dump({"X": X.tolist(), "y": y.tolist()}, f)

    print(f"  Features cached to {cache_path}")
    return X, y


# ── Train & evaluate ──────────────────────────────────────────────────────────

def train_and_evaluate(X: np.ndarray, y: np.ndarray):
    print("\n── Training classifiers ─────────────────────────────────────")

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, proba)
        ap = average_precision_score(y, proba)
        auc = roc_auc_score(y, proba)

        # Find best threshold by F1
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        preds = (proba >= best_thresh).astype(int)
        report = classification_report(y, preds, output_dict=True)

        results[name] = {
            "average_precision": round(ap, 4),
            "roc_auc": round(auc, 4),
            "best_threshold": round(float(best_thresh), 4),
            "precision_hallucination": round(report["1"]["precision"], 4),
            "recall_hallucination": round(report["1"]["recall"], 4),
            "f1_hallucination": round(report["1"]["f1-score"], 4),
        }

        ax.plot(recall, precision, label=f"{name} (AP={ap:.2f})", linewidth=2)
        print(f"\n  {name}:")
        print(f"    AP={ap:.3f}  ROC-AUC={auc:.3f}  "
              f"P={results[name]['precision_hallucination']:.3f}  "
              f"R={results[name]['recall_hallucination']:.3f}")

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve — Hallucination Detection", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="P=0.8 target")
    ax.axvline(x=0.7, color="gray", linestyle=":", alpha=0.5, label="R=0.7 target")
    fig.tight_layout()
    fig.savefig("reports/pr_curve.png", dpi=150)
    print("\n  PR curve saved to reports/pr_curve.png")

    with open("reports/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Metrics saved to reports/metrics.json")

    return results


# ── Save best model ───────────────────────────────────────────────────────────

def save_best_model(X: np.ndarray, y: np.ndarray, results: dict):
    """Train the best model on ALL data and save it."""
    best_name = max(results, key=lambda k: results[k]["average_precision"])
    print(f"\n── Saving best model: {best_name} ──────────────────────────")

    if best_name == "LogisticRegression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
        ])

    model.fit(X, y)
    joblib.dump(model, "models/hallucination_model.joblib")

    meta = {
        "model_name": best_name,
        "feature_names": FEATURE_NAMES,
        "best_threshold": results[best_name]["best_threshold"],
        "metrics": results[best_name],
    }
    with open("models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Model saved to models/hallucination_model.joblib")
    print(f"  Metadata saved to models/model_meta.json")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start = time.time()
    print("=== LLM Hallucination Classifier Training ===\n")
    print("Step 1/3 — Building feature matrix")
    X, y = build_feature_matrix(SAMPLES)
    print(f"  Matrix shape: {X.shape}  |  Hallucination rate: {y.mean():.1%}")

    print("\nStep 2/3 — Training & cross-validation")
    results = train_and_evaluate(X, y)

    print("\nStep 3/3 — Saving best model")
    save_best_model(X, y, results)

    elapsed = time.time() - start
    print(f"\n=== Done in {elapsed:.1f}s ===")
    print("\nNext steps:")
    print("  • Run the API:        uvicorn api.main:app --reload")
    print("  • Run the dashboard:  streamlit run dashboard/app.py")
