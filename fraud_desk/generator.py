"""Synthetic card-transaction generator.

Fraud usually stacks several warning signals, but not always: some fraud shows a single
strong signal, and some genuine customers trip two (a traveller paying abroad at 3am,
a one-off big purchase on a new phone). That overlap is what makes the game non-trivial
and why the model is probabilistic rather than a rule.

The same distribution is mirrored in the JavaScript version of Fraud Hunt (index.html
on saeed-abass.github.io), so the model trained here is valid for both.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

FRAUD_RATE = 0.24  # far above the real ~0.17%, so a game round has something to catch

MERCH_OK = ["Tesco", "Greggs", "Spotify", "TfL", "Boots", "Sainsbury's", "Amazon UK",
            "Costa", "Uber", "Deliveroo", "Shell", "Morrisons"]
MERCH_RISK = ["GiftCardHub", "CryptoXpress", "LuxWatch Online", "ElectroDeals", "QuickTopUp"]
CARD_PRESENT = {"TfL", "Greggs", "Costa"}
ABROAD = ["NG", "RU", "BR", "VN", "US", "AE"]

FRAUD_SIGNALS = ["amount", "night", "abroad", "velocity", "newdevice", "riskmerch"]
BENIGN_SIGNALS = ["amount", "night", "abroad", "newdevice"]


@dataclass
class Transaction:
    merchant: str
    amount: float
    hour: int
    minute: int
    country: str
    device: str          # "known device" | "new device"
    attempts: int        # attempts in the last 2 minutes
    channel: str         # "card present" | "online"
    is_fraud: bool
    signals: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Transaction":
        return Transaction(**d)


def _pick_distinct(rng: random.Random, pool: list[str], n: int) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        s = rng.choice(pool)
        if s not in out:
            out.append(s)
    return out


def make_transaction(rng: random.Random) -> Transaction:
    fraud = rng.random() < FRAUD_RATE
    x = rng.random()
    if fraud:
        if x < 0.22:
            signals = [rng.choice(["velocity", "riskmerch"])]
        else:
            signals = _pick_distinct(rng, FRAUD_SIGNALS, 2 if x < 0.65 else 3)
    else:
        n = 0 if x < 0.55 else 1 if x < 0.87 else 2
        signals = _pick_distinct(rng, BENIGN_SIGNALS, n)

    if "amount" in signals:
        amount = rng.uniform(640, 2400)
    else:
        amount = math.exp(rng.uniform(math.log(3), math.log(140)))
    hour = rng.randint(1, 4) if "night" in signals else rng.randint(7, 22)
    country = rng.choice(ABROAD) if "abroad" in signals else "UK"
    merchant = rng.choice(MERCH_RISK) if "riskmerch" in signals else rng.choice(MERCH_OK)
    attempts = rng.randint(4, 8) if "velocity" in signals else 1
    device = "new device" if "newdevice" in signals else "known device"
    channel = "card present" if (merchant in CARD_PRESENT and not fraud) else "online"

    return Transaction(merchant=merchant, amount=round(amount, 2), hour=hour,
                       minute=rng.randint(0, 59), country=country, device=device,
                       attempts=attempts, channel=channel, is_fraud=fraud, signals=signals)


def make_deck(seed: int, n: int = 500) -> list[Transaction]:
    rng = random.Random(seed)
    return [make_transaction(rng) for _ in range(n)]


SIGNAL_TEXT = {
    "amount": "unusually large amount",
    "night": "made between 1am and 5am",
    "abroad": "card used abroad",
    "velocity": "several attempts in two minutes",
    "newdevice": "new device",
    "riskmerch": "high-risk merchant category",
}


def explain(tx: Transaction) -> str:
    if not tx.signals:
        return "no warning signals"
    return ", ".join(SIGNAL_TEXT[s] for s in tx.signals)
