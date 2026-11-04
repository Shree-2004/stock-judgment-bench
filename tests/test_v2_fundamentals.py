"""The v2_fundamentals prompt and its L1 check (fundamentals track, step 3).

What is under test:
- v1's text is untouched: v2 is v1 plus one block and one contract clause
- a version can't silently gain or lose the block
- results figures a model quotes are parsed and checked exactly like prices
- the block shown is recorded with the brief; v1 briefs keep their shape
"""
from __future__ import annotations

import hashlib
from datetime import date
from types import SimpleNamespace

import pytest

from evals.number_check import check_numbers
from live.fundamentals import Fundamentals, Quarter
from llm import client
from llm.brief import parse_brief
from llm.prompts import _CONTRACT, V1_STRUCTURED, build_prompt


class Snap:
    ticker, company_name, currency, as_of = "HEROMOTOCO.NS", "Hero MotoCorp", "INR", "2026-11-02"

    def figures(self):
        return {"price": 5168.2, "change_pct_1d": 0.42}


class Disc:
    items = []

    def as_prompt_block(self):
        return "(none)"


def _q(end, revenue, profit, eps, published="2026-10-28T15:38:16+00:00"):
    return Quarter(date.fromisoformat(end), published, "revenue_from_operations", revenue * 1e7, profit * 1e7, eps)


HERO = Fundamentals("HEROMOTOCO", "Consolidated", (
    _q("2026-09-30", 14970.50, 1860.55, 92.99), _q("2026-06-30", 12000, 1300, 65.0),
    _q("2026-03-31", 11000, 1200, 60.0), _q("2025-12-31", 10500, 1100, 55.0),
    _q("2025-09-30", 12218.80, 1320.94, 66.0)), close=5168.2)

V2_ANSWER = """VERDICT: BULLISH
CONFIDENCE: 3
KEY_FIGURES:
- price: 5168.20
- net_profit_q_cr: 1,860.55
- net_profit_yoy_pct: 99.99
DRIVERS: none
BULL_CASE: Profit grew.
BEAR_CASE: Valuation."""


def test_v1_text_is_byte_identical_to_what_the_running_tracks_use():
    text = V1_STRUCTURED.system + V1_STRUCTURED.template + _CONTRACT
    assert hashlib.sha256(text.encode()).hexdigest() == \
        "575276e116a016d9ddacd4c54d87281ac606a6910ac3f861c304a8678b34551c"


def test_v2_is_v1_plus_the_results_block():
    _, v1 = build_prompt(Snap(), Disc(), "v1_structured")
    _, v2 = build_prompt(Snap(), Disc(), "v2_fundamentals", HERO)
    assert "QUARTERLY RESULTS" not in v1
    assert "QUARTERLY RESULTS filed with NSE before 2026-11-02" in v2
    assert "Latest results: Q2 FY27 (quarter ended 30 Sep 2026), consolidated, published 28 Oct 2026" in v2
    assert "- net_profit_q_cr: 1,860.55" in v2 and "- pe_4q: " in v2
    assert "copied from MARKET DATA or QUARTERLY\n  RESULTS" in v2
    assert v2.index("MARKET DATA") < v2.index("QUARTERLY RESULTS") < v2.index("DISCLOSURES filed")


@pytest.mark.parametrize("version,fundamentals", [("v2_fundamentals", None), ("v1_structured", HERO),
                                                  ("v2_prose", HERO)])
def test_a_version_cannot_gain_or_lose_the_block(version, fundamentals):
    with pytest.raises(ValueError):
        build_prompt(Snap(), Disc(), version, fundamentals)


def test_quoted_results_are_parsed_and_checked_like_prices():
    truth = {**Snap().figures(), **HERO.figures()}
    l1 = check_numbers(parse_brief(V2_ANSWER).cited_figures, truth)
    assert {c.label: c.status for c in l1.checks} == {
        "price": "matched", "net_profit_q_cr": "matched", "net_profit_yoy_pct": "wrong"}


def test_every_fundamentals_label_survives_the_brief_parser():
    figs = HERO.figures()
    text = "VERDICT: NEUTRAL\nKEY_FIGURES:\n" + "\n".join(f"- {k}: {v:,.2f}" for k, v in figs.items())
    assert parse_brief(text).cited_figures == pytest.approx(figs, abs=0.005)


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(client, "settings", SimpleNamespace(llm_model="m", llm_temperature=0.0,
                                                            number_match_tolerance=0.01))
    monkeypatch.setattr(client, "call_llm", lambda system, user, **kw: (V2_ANSWER, {"total_tokens": 1}))


def test_generate_brief_checks_results_figures_and_records_the_block(fake_llm):
    out = client.generate_brief(Snap(), Disc(), prompt_version="v2_fundamentals", trace=False, fundamentals=HERO)
    d = out.as_dict()
    assert d["l1"]["checks"][1]["status"] == "matched" and not out.l1.clean     # the 99.99 is caught
    assert d["fundamentals"]["quarter_end"] == "2026-09-30"
    assert d["fundamentals"]["figures"] == HERO.figures()
    assert d["fundamentals"]["block"].startswith("Latest results: Q2 FY27")


def test_v1_briefs_keep_their_exact_shape(fake_llm):
    d = client.generate_brief(Snap(), Disc(), prompt_version="v1_structured", trace=False).as_dict()
    assert "fundamentals" not in d
    assert {c["label"]: c["status"] for c in d["l1"]["checks"]}["net_profit_q_cr"] == "unverifiable"
