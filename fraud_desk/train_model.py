"""Train the Fraud Desk model on synthetic transactions and write model.json.

    pip install scikit-learn
    python -m fraud_desk.train_model

Uses a stratified train/validation/test split, standardised features, class-weighted
logistic regression, and picks the decision threshold that maximises F1 on validation
(not on test), then reports test metrics.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .generator import make_deck
from .model import FEATURES, MODEL_PATH, features


def main(seed: int = 7, n: int = 30_000) -> None:
    deck = make_deck(seed, n)
    X = np.array([features(t) for t in deck])
    y = np.array([t.is_fraud for t in deck], dtype=int)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(scaler.transform(X_tr), y_tr)

    p_va = clf.predict_proba(scaler.transform(X_va))[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    threshold = float(max(grid, key=lambda t: f1_score(y_va, p_va >= t)))

    p_te = clf.predict_proba(scaler.transform(X_te))[:, 1]
    pred = p_te >= threshold
    metrics = {
        "precision": round(float(precision_score(y_te, pred)), 3),
        "recall": round(float(recall_score(y_te, pred)), 3),
        "f1": round(float(f1_score(y_te, pred)), 3),
        "pr_auc": round(float(average_precision_score(y_te, p_te)), 3),
        "test_size": int(len(y_te)),
    }
    out = {
        "features": FEATURES,
        "weights": [round(float(w), 6) for w in clf.coef_[0]],
        "bias": round(float(clf.intercept_[0]), 6),
        "mean": [round(float(m), 6) for m in scaler.mean_],
        "scale": [round(float(s), 6) for s in scaler.scale_],
        "threshold": round(threshold, 3),
        "meta": {"algorithm": "logistic regression (class-weighted)", "trained_on": f"{n} synthetic transactions, seed {seed}",
                 "test_metrics": metrics},
    }
    MODEL_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["meta"], indent=2))
    print(dict(zip(FEATURES, out["weights"])))


if __name__ == "__main__":
    main()
