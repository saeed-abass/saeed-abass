"""Logistic-regression fraud scorer. Pure Python at inference time (no dependencies),
so the GitHub Action stays fast. Weights come from train_model.py."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .generator import MERCH_RISK, Transaction

MODEL_PATH = Path(__file__).with_name("model.json")

FEATURES = ["log_amount", "night", "abroad", "log_attempts", "new_device",
            "risky_merchant", "online"]


def features(tx: Transaction) -> list[float]:
    return [
        math.log1p(tx.amount),
        1.0 if tx.hour < 5 else 0.0,
        1.0 if tx.country != "UK" else 0.0,
        math.log(tx.attempts),
        1.0 if tx.device == "new device" else 0.0,
        1.0 if tx.merchant in MERCH_RISK else 0.0,
        1.0 if tx.channel == "online" else 0.0,
    ]


class Model:
    def __init__(self, weights: list[float], bias: float, mean: list[float],
                 scale: list[float], threshold: float, meta: dict | None = None):
        self.weights, self.bias = weights, bias
        self.mean, self.scale = mean, scale
        self.threshold = threshold
        self.meta = meta or {}

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "Model":
        d = json.loads(Path(path).read_text())
        return cls(d["weights"], d["bias"], d["mean"], d["scale"], d["threshold"], d.get("meta"))

    def proba(self, tx: Transaction) -> float:
        z = self.bias
        for w, x, m, s in zip(self.weights, features(tx), self.mean, self.scale):
            z += w * (x - m) / s
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, tx: Transaction) -> bool:
        return self.proba(tx) >= self.threshold
