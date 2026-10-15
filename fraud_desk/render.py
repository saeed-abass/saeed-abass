"""Render the Fraud Desk board into README.md between the START/END markers."""
from __future__ import annotations

import re
from pathlib import Path

from .engine import Links, State, metrics
from .model import Model

START, END = "<!-- FRAUD-DESK:START -->", "<!-- FRAUD-DESK:END -->"


def _pct(x: float, n: int = 1) -> str:
    return f"{x:.0%}" if n else "n/a"


def board(state: State, model: Model, links: Links) -> str:
    tx = state.current()
    c, m = metrics(state.community), metrics(state.model)
    lines = [
        START,
        f"**Case #{state.case_no}** has just landed on the desk. Fraud or genuine? Make the call:",
        "",
        "| 🏪 Merchant | 💷 Amount | 🕒 Time | 🌍 Country | 📱 Device | 🔁 Attempts (2 min) | 🛒 Channel |",
        "|---|--:|:-:|:-:|---|:-:|---|",
        f"| {tx.merchant} | £{tx.amount:,.2f} | {tx.hour:02d}:{tx.minute:02d} | {tx.country} | {tx.device} | {tx.attempts} | {tx.channel} |",
        "",
        f"<p align=\"center\"><a href=\"{links.move('flag', state.case_no)}\"><b>🚩 FLAG AS FRAUD</b></a>"
        f" &nbsp;&nbsp;·&nbsp;&nbsp; "
        f"<a href=\"{links.move('pass', state.case_no)}\"><b>✅ LET IT THROUGH</b></a></p>",
        "",
        "<sub>Clicking opens a pre-filled GitHub issue: press <b>Create</b>. A GitHub Actions workflow checks your call, "
        "scores my model on the same case, replies on the issue and updates this board within about a minute.</sub>",
        "",
        "| | 👥 Everyone so far | 🤖 My model, same cases |",
        "|---|:-:|:-:|",
        f"| Cases decided | {c['cases']} | {m['cases']} |",
        f"| Precision | {_pct(c['precision'], c['cases'])} | {_pct(m['precision'], m['cases'])} |",
        f"| Recall | {_pct(c['recall'], c['cases'])} | {_pct(m['recall'], m['cases'])} |",
        f"| F1 | **{c['f1']:.2f}** | **{m['f1']:.2f}** |" if c["cases"] else "| F1 | n/a | n/a |",
        "",
    ]
    ranked = sorted(
        ((login, metrics(p["cm"]), p["best_streak"]) for login, p in state.players.items()),
        key=lambda r: (r[1]["accuracy"] * min(r[1]["cases"], 10), r[1]["cases"], r[2]), reverse=True)[:5]
    if ranked:
        lines += ["<details><summary><b>🏆 Leaderboard</b> and recent calls</summary>", "",
                  "| # | Detective | Cases | Correct | Best streak |", "|:-:|---|:-:|:-:|:-:|"]
        medals = ["🥇", "🥈", "🥉", "4", "5"]
        lines += [f"| {medals[i]} | @{login} | {mm['cases']} | {_pct(mm['accuracy'])} | {streak} |"
                  for i, (login, mm, streak) in enumerate(ranked)]
        lines += ["", "| Case | Player | Call | Truth | Model |", "|:-:|---|---|---|:-:|"]
        lines += [f"| #{r['case']} | @{r['login']} | {r['verdict']} {'✔️' if r['correct'] else '✖️'} | {r['truth']} | "
                  f"{'✔️' if r['model_correct'] else '✖️'} |" for r in state.recent]
        lines += ["", "</details>", ""]
    else:
        lines += ["<sub>🏆 No calls yet. Be the first detective on the leaderboard.</sub>", ""]
    lines += [
        "<details><summary><b>⚙️ How this works</b></summary>", "",
        "1. Each case is a synthetic card transaction. Fraud usually stacks warning signals, but some fraud shows only one, "
        "and some genuine customers trip two, so no simple rule gets everything right.",
        f"2. My model is a class-weighted logistic regression trained on 30,000 synthetic transactions "
        f"(held-out test F1 {model.meta.get('test_metrics', {}).get('f1', 0):.2f}), with its threshold tuned on a "
        "separate validation set. It sees exactly what you see.",
        f"3. Your click opens an issue; [a workflow]({links.repo}/blob/main/.github/workflows/fraud-desk.yml) runs "
        f"[the Python engine]({links.repo}/tree/main/fraud_desk), scores you and the model, commits this board and "
        f"replies to you. [Tests]({links.repo}/tree/main/tests) run in CI on every push.",
        "4. For the real thing on real data, see my "
        "[fraud detection project](https://github.com/saeed-abass/credit-card-fraud-detection-mlflow) (XGBoost + MLflow).",
        "", "</details>", END,
    ]
    return "\n".join(lines)


def write_readme(path: Path, state: State, model: Model, links: Links) -> bool:
    text = path.read_text()
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise ValueError(f"README is missing the {START} / {END} markers")
    new = pattern.sub(lambda _: board(state, model, links), text)
    if new != text:
        path.write_text(new)
        return True
    return False
