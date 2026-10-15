"""Command-line entry point used by the GitHub Actions workflow.

    python -m fraud_desk play  --issues issues.json --out outcomes/
    python -m fraud_desk render

``issues.json`` is the output of
``gh issue list --state open --search "fraud-desk in:title" --json number,title,author,createdAt``.
Every open move issue is processed in creation order, so if GitHub drops a queued
workflow run under load, the next run still catches up. For each processed issue the
command writes ``<out>/<number>.md`` (the reply) and lists it in ``<out>/close.txt``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .engine import Links, State, apply_move
from .model import Model
from .render import write_readme

ROOT = Path(__file__).resolve().parent.parent


def links_from_env() -> Links:
    owner, _, name = os.environ.get("GITHUB_REPOSITORY", "saeed-abass/saeed-abass").partition("/")
    return Links(owner, name)


def main() -> None:
    ap = argparse.ArgumentParser(prog="fraud_desk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("--issues", required=True)
    p.add_argument("--out", required=True)
    sub.add_parser("render")
    args = ap.parse_args()

    state, model, links = State.load(), Model.load(), links_from_env()

    if args.cmd == "play":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        issues = json.loads(Path(args.issues).read_text())
        issues.sort(key=lambda i: (i.get("createdAt", ""), i["number"]))
        to_close = []
        for issue in issues:
            author = (issue.get("author") or {})
            if author.get("is_bot") or author.get("login", "").endswith("[bot]"):
                continue
            o = apply_move(state, model, int(issue["number"]), issue.get("title", ""),
                           author.get("login", "someone"), links)
            if o.comment:
                (out / f"{o.issue}.md").write_text(o.comment)
                to_close.append(str(o.issue))
        (out / "close.txt").write_text("\n".join(to_close))
        state.save()
        print(f"processed {len(to_close)} move(s); case #{state.case_no} on the desk")

    write_readme(ROOT / "README.md", state, model, links)


if __name__ == "__main__":
    main()
