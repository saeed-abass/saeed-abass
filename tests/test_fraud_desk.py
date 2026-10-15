import json
import random
from pathlib import Path

import pytest

from fraud_desk.engine import Links, State, apply_move, metrics, parse_title
from fraud_desk.generator import make_deck, make_transaction
from fraud_desk.model import Model
from fraud_desk.render import END, START, board, write_readme

LINKS = Links("saeed-abass", "saeed-abass")


@pytest.fixture
def model():
    return Model.load()


def fresh_state():
    return State(seed=1)


# ---- generator -------------------------------------------------------------

def test_deck_is_deterministic():
    a, b = make_deck(3, 50), make_deck(3, 50)
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]


def test_fraud_rate_and_overlap():
    deck = make_deck(11, 5000)
    rate = sum(t.is_fraud for t in deck) / len(deck)
    assert 0.20 < rate < 0.28
    # the game must not be solvable by "two or more signals means fraud"
    assert any(not t.is_fraud and len(t.signals) >= 2 for t in deck)
    assert any(t.is_fraud and len(t.signals) == 1 for t in deck)


def test_card_present_is_never_fraud():
    deck = make_deck(5, 3000)
    assert all(t.channel == "online" for t in deck if t.is_fraud)


# ---- model -----------------------------------------------------------------

def test_model_is_strong_but_not_perfect(model):
    deck = make_deck(999, 4000)   # seed never used in training
    cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for t in deck:
        pred = model.predict(t)
        cm[("tp" if t.is_fraud else "fp") if pred else ("fn" if t.is_fraud else "tn")] += 1
    m = metrics(cm)
    assert 0.80 < m["f1"] < 0.99


def test_probabilities_are_valid(model):
    rng = random.Random(0)
    for _ in range(200):
        p = model.proba(make_transaction(rng))
        assert 0.0 < p < 1.0


# ---- title parsing ---------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("fraud-desk|flag|12", ("flag", 12)),
    ("fraud-desk|pass|1", ("pass", 1)),
    ("  fraud-desk|flag|7  ", ("flag", 7)),
])
def test_parse_valid(title, expected):
    assert parse_title(title) == expected


@pytest.mark.parametrize("title", [
    "fraud-desk|delete|1", "fraud-desk|flag|", "fraud-desk|flag|1; rm -rf /",
    "Fraud-Desk|flag|1", "fraud-desk|flag|1234567", "hello",
])
def test_parse_rejects_junk(title):
    assert parse_title(title) is None


# ---- moves -----------------------------------------------------------------

def test_correct_move_scores_and_advances(model):
    s = fresh_state()
    tx = s.current()
    action = "flag" if tx.is_fraud else "pass"
    o = apply_move(s, model, 101, f"fraud-desk|{action}|1", "alice", LINKS)
    assert o.accepted and "Correct" in o.comment
    assert s.case_no == 2 and s.index == 1
    assert metrics(s.community)["accuracy"] == 1.0
    assert s.players["alice"]["streak"] == 1
    assert metrics(s.model)["cases"] == 1


def test_wrong_move_resets_streak(model):
    s = fresh_state()
    s.players["bob"] = {"cm": {"tp": 0, "fp": 0, "fn": 0, "tn": 3}, "streak": 3, "best_streak": 3}
    tx = s.current()
    action = "pass" if tx.is_fraud else "flag"
    o = apply_move(s, model, 102, f"fraud-desk|{action}|1", "bob", LINKS)
    assert o.accepted and "Not quite" in o.comment
    assert s.players["bob"]["streak"] == 0 and s.players["bob"]["best_streak"] == 3


def test_stale_case_not_counted(model):
    s = fresh_state()
    apply_move(s, model, 1, "fraud-desk|flag|1", "alice", LINKS)
    o = apply_move(s, model, 2, "fraud-desk|flag|1", "carol", LINKS)
    assert not o.accepted and "already decided" in o.comment
    assert "carol" not in s.players and s.case_no == 2


def test_same_issue_never_processed_twice(model):
    s = fresh_state()
    apply_move(s, model, 5, "fraud-desk|flag|1", "alice", LINKS)
    o = apply_move(s, model, 5, "fraud-desk|flag|2", "alice", LINKS)
    assert o.comment == "" and s.case_no == 2


def test_deck_rolls_over(model):
    s = fresh_state()
    s.index = 499
    s.case_no = 500
    apply_move(s, model, 9, "fraud-desk|flag|500", "dave", LINKS)
    assert s.seed == 2 and s.index == 0 and s.case_no == 501


# ---- rendering -------------------------------------------------------------

def test_board_has_move_links_for_current_case(model):
    s = fresh_state()
    md = board(s, model, LINKS)
    assert md.startswith(START) and md.endswith(END)
    assert "fraud-desk%7Cflag%7C1" in md and "fraud-desk%7Cpass%7C1" in md
    assert "Case #1" in md


def test_write_readme_replaces_only_board(tmp_path, model):
    readme = tmp_path / "README.md"
    readme.write_text(f"# Hi\n\n{START}\nOLD_BOARD_TEXT\n{END}\n\nfooter\n")
    assert write_readme(readme, fresh_state(), model, LINKS)
    text = readme.read_text()
    assert text.startswith("# Hi") and text.endswith("footer\n") and "OLD_BOARD_TEXT" not in text


def test_write_readme_requires_markers(tmp_path, model):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here")
    with pytest.raises(ValueError):
        write_readme(readme, fresh_state(), model, LINKS)


def test_state_roundtrip(tmp_path, model):
    s = fresh_state()
    apply_move(s, model, 1, "fraud-desk|flag|1", "alice", LINKS)
    p = tmp_path / "state.json"
    s.save(p)
    s2 = State.load(p)
    assert s2.case_no == s.case_no and s2.players == s.players
    json.loads(p.read_text())
