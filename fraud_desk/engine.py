"""Game state and move processing for Fraud Desk.

A move is a GitHub issue titled ``fraud-desk|flag|<case>`` or ``fraud-desk|pass|<case>``.
Each valid move on the current case scores the player, scores the model on the same
case, then puts the next case on the board.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .generator import Transaction, explain, make_deck
from .model import Model

STATE_PATH = Path(__file__).with_name("state.json")
TITLE_RE = re.compile(r"^fraud-desk\|(flag|pass)\|(\d{1,6})$")
DECK_SIZE = 500
RECENT_MAX = 6


def parse_title(title: str) -> tuple[str, int] | None:
    m = TITLE_RE.match(title.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2))


def empty_cm() -> dict:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def score(cm: dict, said_fraud: bool, is_fraud: bool) -> None:
    key = ("tp" if is_fraud else "fp") if said_fraud else ("fn" if is_fraud else "tn")
    cm[key] += 1


def metrics(cm: dict) -> dict:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    n = tp + fp + fn + tn
    return {"precision": p, "recall": r, "f1": f1, "cases": n, "accuracy": (tp + tn) / n if n else 0.0}


@dataclass
class State:
    seed: int = 1
    index: int = 0              # position in the current deck
    case_no: int = 1            # public, ever-increasing case number
    community: dict = field(default_factory=empty_cm)
    model: dict = field(default_factory=empty_cm)
    players: dict = field(default_factory=dict)   # login -> {"cm": {...}, "streak": int, "best_streak": int}
    recent: list = field(default_factory=list)
    processed_issues: list = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> "State":
        if not Path(path).exists():
            return cls()
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: Path = STATE_PATH) -> None:
        self.processed_issues = self.processed_issues[-500:]
        Path(path).write_text(json.dumps(self.__dict__, indent=1))

    def current(self) -> Transaction:
        return make_deck(self.seed, DECK_SIZE)[self.index]

    def advance(self) -> None:
        self.index += 1
        self.case_no += 1
        if self.index >= DECK_SIZE:
            self.seed += 1
            self.index = 0


@dataclass
class Outcome:
    issue: int
    comment: str
    accepted: bool


def apply_move(state: State, model: Model, issue: int, title: str, login: str,
               links: "Links", now: datetime | None = None) -> Outcome:
    now = now or datetime.now(timezone.utc)
    if issue in state.processed_issues:
        return Outcome(issue, "", False)
    state.processed_issues.append(issue)

    parsed = parse_title(title)
    if parsed is None:
        return Outcome(issue, f"I couldn't read that move, so it wasn't counted. Case **#{state.case_no}** is on the desk: "
                              f"{links.play_line(state.case_no)}", False)
    action, case = parsed
    if case != state.case_no:
        return Outcome(issue, (
            f"Case #{case} was already decided by another player, so this move wasn't counted. "
            f"Case **#{state.case_no}** is on the desk now: {links.play_line(state.case_no)}"), False)

    tx = state.current()
    said_fraud = action == "flag"
    model_p = model.proba(tx)
    model_fraud = model_p >= model.threshold
    correct = said_fraud == tx.is_fraud

    score(state.community, said_fraud, tx.is_fraud)
    score(state.model, model_fraud, tx.is_fraud)
    p = state.players.setdefault(login, {"cm": empty_cm(), "streak": 0, "best_streak": 0})
    score(p["cm"], said_fraud, tx.is_fraud)
    p["streak"] = p["streak"] + 1 if correct else 0
    p["best_streak"] = max(p["best_streak"], p["streak"])

    truth = "fraud" if tx.is_fraud else "genuine"
    verdict = "🚩 flagged" if said_fraud else "✅ passed"
    state.recent.insert(0, {
        "case": state.case_no, "login": login, "verdict": verdict, "truth": truth,
        "correct": correct, "model_correct": model_fraud == tx.is_fraud, "at": now.strftime("%d %b %H:%M UTC"),
    })
    del state.recent[RECENT_MAX:]

    pm = metrics(p["cm"])
    prob = ">0.99" if model_p > 0.99 else "<0.01" if model_p < 0.01 else f"{model_p:.2f}"
    model_line = (f"My model said **{'fraud' if model_fraud else 'genuine'}** "
                  f"(fraud probability {prob}), so it was {'right' if model_fraud == tx.is_fraud else 'wrong'} too."
                  if (model_fraud == tx.is_fraud) == correct else
                  f"My model said **{'fraud' if model_fraud else 'genuine'}** (fraud probability {prob}), "
                  f"so {'you beat it on this one' if correct else 'it got this one and you did not'}.")
    comment = "\n".join([
        f"### {'✅ Correct' if correct else '❌ Not quite'}, @{login}",
        f"Case #{state.case_no} ({tx.merchant}, £{tx.amount:,.2f}) was **{truth}**. Signals: {explain(tx)}.",
        "",
        model_line,
        "",
        f"Your record: {pm['cases']} case{'s' if pm['cases'] != 1 else ''}, {pm['accuracy']:.0%} correct, current streak {p['streak']}.",
        "",
        f"Case **#{state.case_no + 1}** is now on the desk: {links.readme}",
        "",
        "<sub>This reply was written by a GitHub Actions workflow running Python. "
        f"[See how it works]({links.repo}/tree/main/fraud_desk).</sub>",
    ])
    state.advance()
    return Outcome(issue, comment, True)


@dataclass
class Links:
    owner: str
    repo_name: str

    @property
    def repo(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo_name}"

    @property
    def readme(self) -> str:
        return f"https://github.com/{self.owner}"   # the profile page shows this README

    def move(self, action: str, case: int) -> str:
        body = ("Just press **Create** (or Submit new issue). A GitHub Action will check your answer, reply here "
                "within about a minute, close this issue and update the board on my profile.")
        from urllib.parse import quote
        return f"{self.repo}/issues/new?title={quote(f'fraud-desk|{action}|{case}', safe='')}&body={quote(body, safe='')}"

    def play_line(self, case: int) -> str:
        return f"[🚩 flag as fraud]({self.move('flag', case)}) · [✅ let it through]({self.move('pass', case)})"
