"""M2: rule classifier and the held-out label workflow, offline.

The example descriptions are copied verbatim from the live feed (28-29 Oct
2026). They are the cases the rules were written against, so passing them
proves the rules do what they say. It says nothing about accuracy on new
text. That number comes only from the hand-labelled sample.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta

import pytest

import live.classify as classify_mod
from live import store
from live.classify import CATEGORIES, classify, normalise_tag, rules_fingerprint, split_subject
from live.filings import FilingRecord, provider_ref
from scripts import label_filings

S = " |SUBJECT: "

# (description, expected category, expected rule)
CASES = [
    # decided by the tag
    ("HDFC Asset Management Company Limited has informed the Exchange that the Net Asset Value (per unit) "
     "of HDFC Mutual Fund - HDFC NIFTY 100 ETF as on October 28, 2026 is Rs. 24.47." + S + "Declaration of NAV",
     "fund_nav", "tag.nav"),
    ("Balrampur Chini Mills Limited has informed the Exchange about Board Meeting to be held on 04-Nov-2026 "
     "to consider and approve the Quarterly Unaudited Financial results" + S + "Board Meeting Intimation",
     "board_meeting", "tag.bm_notice"),
    ("Infosys Limited has informed the Exchange regarding 'Earnings Call Transcript'." + S
     + "Analysts/Institutional Investor Meet/Con. Call Updates", "investor_comms", "tag.investor"),
    ("Kiri Industries has informed the Exchange about Acquisition" + S
     + "Acquisition (including agreement to acquire)-XBRL", "deal", "tag.deal"),
    ("x" + S + "Bagging/Receiving of orders/contracts  (Sub-para 4-Para B)", "business_update", "tag.business"),
    ("x" + S + "Change in Directors/KMP/SMP/Auditor/RTA", "management_change", "tag.management"),
    ("x" + S + "Credit Rating- Revision", "credit_rating", "tag.rating"),
    ("x" + S + "Fraud/Default/Arrest", "legal_regulatory", "tag.legal"),
    ("x" + S + "Confirmation of Redemption/Payment of Interest and Principal", "debt_and_proceeds", "tag.debt"),
    ("x" + S + "Shareholders meeting", "shareholder_meeting", "tag.agm"),
    ("Supreme Industries Limited: Newspaper Publication of Q2 Results" + S + "Copy of Newspaper Publication",
     "routine_compliance", "tag.compliance"),
    ("x" + S + "ISD for Open Offer-Pre Tendering Stage", "deal", "tag.open_offer"),
    ("x" + S + "ISD for Buyback-Open Market", "corporate_action", "tag.corp_action"),
    ("x" + S + "ESOP/ESOS/ESPS", "capital_raise", "tag.capital"),
    # generic tag -> free text
    ("Aarti Pharmalabs Limited has submitted to the Exchange, the financial results for the period ended "
     "September 30, 2026." + S + "Outcome of Board Meeting", "results", "text.results"),
    ("Sasken Technologies Limited has informed the Exchange regarding Outcome of Board Meeting held on "
     "28-Oct-2026 for Dividend" + S + "Outcome of Board Meeting-XBRL", "corporate_action", "text.corp_action"),
    ("Prudent Corporate Advisory Services Limited has informed the Exchange regarding a press release dated "
     "October 28, 2026, titled \"Prudent's Operating Profit Rises 42.4% YoY\"." + S + "Press Release",
     "results", "text.results"),
    ("Jupiter Wagons Limited has informed the Exchange about securing order for 2,583 BOXNHL and BOXNS wagons "
     "worth INR 1,140 crore from the Ministry of Railways." + S + "General Updates",
     "business_update", "text.business"),
    ("Engineers India Limited has informed the Exchange regarding 'Board Comments on the Fine imposed by the "
     "Exchange'." + S + "Updates", "legal_regulatory", "text.legal"),
    ("GP Petroleums Limited has informed the Exchange about Investor Presentation for the Q2 FY 2027." + S
     + "General Updates", "investor_comms", "text.investor"),
    ("Brookfield India Real Estate Trust has informed the Exchange regarding 'Intimation of Meeting of Board of "
     "Directors  scheduled to be held on November 02, 2026'." + S + "Updates", "board_meeting", "text.bm_notice"),
    ("Nilkamal Limited has informed the Exchange about Resignation of Senior Management Personnel." + S
     + "General Updates", "management_change", "text.management"),
    ("Paras Healthcare Limited has informed the Exchange regarding 'Code of Practices and Procedures for Fair "
     "Disclosure of UPSI under Regulation 8(2) SEBI PIT Regulations'." + S + "Updates",
     "routine_compliance", "text.compliance"),
    # generic tag, specific text that matches no rule -> the tag's fallback
    ("Karur Vysya Bank Limited has informed the Exchange regarding 'Opening of New Branches'." + S + "Updates",
     "business_update", "fallback.updates"),
    ("Crompton Greaves Consumer Electricals Limited has informed the Exchange regarding a press release dated "
     "October 27, 2026, titled \"Crompton Strengthens Water Heating Portfolio\"." + S + "Press Release",
     "business_update", "fallback.press release"),
    # generic tag, nothing said -> other
    ("Pidilite Industries Limited has informed the Exchange about General Updates" + S + "General Updates",
     "other", "empty"),
    ("Gokul Agro Resources Limited has informed the Exchange regarding a press release dated October 28, 2026, "
     "titled \"Press Release\"." + S + "Press Release", "other", "empty"),
    # an outcome that names no decision is board_meeting by definition (v1 said other)
    ("Rite Zone Chemcon India Limited has informed the Exchange regarding Outcome of Board Meeting held on "
     "October 28, 2026." + S + "Outcome of Board Meeting", "board_meeting", "empty.outcome"),
    ("Outcome of Board Meeting" + S + "Outcome of Board Meeting", "board_meeting", "empty.outcome"),
    # statutory preamble cut off mid-sentence says nothing (v1: business_update)
    ("Bajaj Electricals Limited has informed the Exchange about disclosure under Regulation 30 of Securities and "
     "Exchange Board of India (Listing Obligations and Disclosures Requirements) Regulations, 2015," + S
     + "General Updates", "other", "empty"),
    # (v1: business_update)
    ("CG Power and Industrial Solutions Limited has informed the Exchange regarding a press release dated "
     "October 29, 2026, titled \"Axiro Semiconductor Invests in ThinSiC to Strengthen Position\"." + S
     + "Press Release", "deal", "text.deal"),
    # no tag at all (price-movement notices)
    ("Significant movement in price has been observed in R.K. Fashion Accessories Limited.",
     "routine_compliance", "text.compliance"),
]


@pytest.mark.parametrize("subject,category,rule", CASES, ids=[f"{c}:{r}" for _, c, r in CASES])
def test_real_descriptions(subject, category, rule):
    got = classify(subject)
    assert (got.category, got.rule) == (category, rule)


# Empty description, content only in the PDF filename — the v1 failure mode.
# Real filenames from the v1 sample; the descriptions are exactly as filed.
P = "https://nsearchives.nseindia.com/corporate/"
URL_CASES = [
    ("Updates" + S + "Updates", P + "esfb_29102026135945_SEIntimation_Investor_Presentation.pdf",
     "investor_comms", "url.investor"),
    ("Updates" + S + "Updates", P + "IDFCIDFL_29102026142218_NIIF_IFL_Outcome_of_BM_29102026_sign.pdf",
     "board_meeting", "url.bm"),
    ("Srivasavi Adhesive Tapes Limited has informed the Exchange regarding Outcome of Board Meeting held on "
     "October 29, 2026." + S + "Outcome of Board Meeting", P + "SRIVASAVI_29102026142254_BM_Outcome_for_CFO.pdf",
     "management_change", "url.management"),
    ("Wheels India Limited has informed the Exchange regarding a press release dated October 29, 2026, titled "
     "\"Press Release\"." + S + "Press Release", P + "WHEELS_29102026144308_WILReg30PressreleaseQ2FY202627.pdf",
     "results", "url.results"),
    ("Atul Limited has informed the Exchange about General Updates" + S + "General Updates",
     P + "ATUL1_29102026140840_1_AL_-_NSE_Buyback_Undertaking.pdf", "corporate_action", "url.corp_action"),
    # the filename says nothing either -> unchanged
    ("Popular Vehicles and Services Limited has informed the Exchange regarding Outcome of Board Meeting held on "
     "October 29, 2026." + S + "Outcome of Board Meeting", P + "9072530155_29102026130826_IntimationSE_SD.pdf",
     "board_meeting", "empty.outcome"),
    ("Pidilite Industries Limited has informed the Exchange about General Updates" + S + "General Updates",
     None, "other", "empty"),
]


@pytest.mark.parametrize("subject,url,category,rule", URL_CASES, ids=[r for *_, r in URL_CASES])
def test_filename_is_read_only_when_the_description_is_empty(subject, url, category, rule):
    got = classify(subject, url)
    assert (got.category, got.rule) == (category, rule)


def test_filename_never_overrides_a_description_that_says_something():
    subject = ("Jupiter Wagons Limited has informed the Exchange about securing order for 2,583 wagons worth "
               "INR 1,140 crore from the Ministry of Railways." + S + "General Updates")
    assert classify(subject, P + "JWL_29102026_Investor_Presentation.pdf").category == "business_update"


def test_url_words_handles_missing_and_nan():
    from live.classify import url_words
    assert url_words(None) == "" and url_words(float("nan")) == ""
    assert url_words(P + "WHEELS_29102026144308_WILReg30PressreleaseQ2FY202627.pdf") == \
        "wilreg30pressrelease q2fy202627"          # squashed words stay squashed; URL_RULES allow for it


def test_every_category_except_other_has_a_case():
    covered = {c for _, c, _ in CASES}
    assert set(CATEGORIES) - covered == set()


def test_tag_variants_normalise_to_one_tag():
    assert normalise_tag(" Bagging/Receiving of orders/contracts  (Sub-para 4-Para B)") == \
        normalise_tag("Bagging/Receiving of orders/contracts")
    assert normalise_tag("Outcome of Board Meeting-XBRL") == "outcome of board meeting"
    assert normalise_tag("Update-Acquisition (including agreement to acquire)-XBRL") == \
        normalise_tag("Acquisition (including agreement to acquire)")
    assert normalise_tag("Pendency of any litigation(s) or dispute(s)-(Sub-para 8-Para B)") == \
        "pendency of any litigation(s) or dispute(s)"


def test_split_subject_without_a_tag():
    assert split_subject("just text") == ("just text", "")
    assert split_subject("") == ("", "")


def test_unknown_specific_tag_is_flagged_not_guessed():
    got = classify("x" + S + "Some Tag NSE Adds Next Year")
    assert (got.category, got.rule) == ("other", "unmatched_tag")


def test_fingerprint_moves_when_a_rule_moves(monkeypatch):
    before = rules_fingerprint()
    monkeypatch.setattr(classify_mod, "TEXT_RULES", classify_mod.TEXT_RULES[:-1])
    assert rules_fingerprint() != before


# ---------------------------------------------------------------------------
# label workflow
# ---------------------------------------------------------------------------

FROZEN = classify_mod.RULES_FROZEN_AT
_F = datetime.fromisoformat(FROZEN)
BEFORE, AFTER = (_F - timedelta(hours=1)).isoformat(), (_F + timedelta(hours=2)).isoformat()


def _rec(i: int, subject: str) -> FilingRecord:
    company, published = f"Company {i} Limited", "2026-10-29T05:00:00+00:00"
    return FilingRecord("nse_rss", provider_ref(company, published, None, subject),
                        company, subject, None, published, "29-Oct-2026 10:30:00")


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A store with 80 eligible filings (30 generic-tag) plus decoys that must
    never be sampled: pre-freeze filings and NAVs."""
    db = tmp_path / "live.sqlite"
    conn = store.connect(db)
    job = store.start_job(conn, "ingest_filings", "nse_rss", BEFORE)
    early = [_rec(i, f"early {i}" + S + "General Updates") for i in range(40)]
    store.insert_filings(conn, early, BEFORE, job)
    later = ([_rec(100 + i, f"Co {i} news item {i}" + S + "General Updates") for i in range(30)]
             + [_rec(200 + i, "x" + S + "Credit Rating") for i in range(50)]
             + [_rec(300 + i, "NAV" + S + "Declaration of NAV") for i in range(40)])
    store.insert_filings(conn, later, AFTER, job)

    for name in ("ROOT", "LABEL_DIR"):
        monkeypatch.setattr(label_filings, name, tmp_path)
    monkeypatch.setattr(label_filings, "SHEET", tmp_path / "sheet.csv")
    monkeypatch.setattr(label_filings, "SAMPLE", tmp_path / "sample.json")
    monkeypatch.setattr(label_filings, "SCORE", tmp_path / "score.json")
    return db, {r.provider_ref: r for r in early + later}


def _sheet(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_sample_is_held_out_and_excludes_navs(lab):
    db, recs = lab
    assert label_filings.main(["--db", str(db), "sample"]) == 0
    rows = _sheet(label_filings.SHEET)
    assert len(rows) == 50
    picked = [recs[r["provider_ref"]] for r in rows]
    assert not any(p.subject.startswith("early") for p in picked)          # pre-freeze
    assert not any("Declaration of NAV" in p.subject for p in picked)       # NAV
    assert all(r["label"] == "" for r in rows)                               # blind

    meta = json.loads(label_filings.SAMPLE.read_text(encoding="utf-8"))
    assert len(meta["strata"]["random"]) == 30 and len(meta["strata"]["hard"]) == 20
    assert all("General Updates" in recs[ref].subject for ref in meta["strata"]["hard"])
    assert meta["rules_fingerprint"] == rules_fingerprint()


def test_sample_is_drawn_once(lab):
    db, _ = lab
    label_filings.main(["--db", str(db), "sample"])
    first = label_filings.SHEET.read_bytes()
    assert label_filings.main(["--db", str(db), "sample"]) == 1
    assert label_filings.SHEET.read_bytes() == first


def test_sample_refuses_when_too_few_held_out(tmp_path, monkeypatch):
    db = tmp_path / "live.sqlite"
    conn = store.connect(db)
    job = store.start_job(conn, "ingest_filings", "nse_rss", BEFORE)
    store.insert_filings(conn, [_rec(i, f"e {i}" + S + "Updates") for i in range(100)], BEFORE, job)
    monkeypatch.setattr(label_filings, "SHEET", tmp_path / "sheet.csv")
    assert label_filings.main(["--db", str(db), "sample"]) == 1
    assert not (tmp_path / "sheet.csv").exists()


def _fill(path, label_for):
    rows = _sheet(path)
    for r in rows:
        r["label"] = label_for(r)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=label_filings.COLUMNS)
        w.writeheader()
        w.writerows(rows)


def test_score_reports_strata_and_disagreements(lab):
    db, _ = lab
    label_filings.main(["--db", str(db), "sample"])
    # the labeller agrees with the rules except on every generic-tag filing
    _fill(label_filings.SHEET,
          lambda r: "other" if "General Updates" in r["description"] else classify(r["description"], r["url"] or None).category)
    assert label_filings.main(["--db", str(db), "score"]) == 0
    score = json.loads(label_filings.SCORE.read_text(encoding="utf-8"))
    assert score["held_out"] is True
    assert score["strata"]["hard"]["correct"] == 0
    assert score["strata"]["all"]["n"] == 50
    assert {(c["gold"], c["predicted"]) for c in score["confusion"]} == {("other", "business_update")}


def test_score_rejects_labels_outside_the_taxonomy(lab):
    db, _ = lab
    label_filings.main(["--db", str(db), "sample"])
    _fill(label_filings.SHEET, lambda r: "earnings")
    assert label_filings.main(["--db", str(db), "score"]) == 1


def test_score_admits_when_rules_changed_after_sampling(lab, monkeypatch):
    db, _ = lab
    label_filings.main(["--db", str(db), "sample"])
    _fill(label_filings.SHEET, lambda r: classify(r["description"], r["url"] or None).category)
    monkeypatch.setattr(classify_mod, "TEXT_RULES", classify_mod.TEXT_RULES[:-1])
    label_filings.main(["--db", str(db), "score"])
    assert json.loads(label_filings.SCORE.read_text(encoding="utf-8"))["held_out"] is False


def test_tool_labels_are_scored_to_a_separate_file_and_marked(lab):
    db, _ = lab
    label_filings.main(["--db", str(db), "sample"])
    tool = label_filings.SHEET.with_name("tool.csv")
    tool.write_bytes(label_filings.SHEET.read_bytes())
    _fill(tool, lambda r: classify(r["description"], r["url"] or None).category)
    assert label_filings.main(["--db", str(db), "score", "--sheet", str(tool), "--labeller", "ai_tool"]) == 0
    assert not label_filings.SCORE.exists()                 # the human score is not overwritten
    out = json.loads(label_filings.SCORE.with_name("score_ai_tool.json").read_text(encoding="utf-8"))
    assert out["labeller"] == "ai_tool"


def test_score_refuses_a_sheet_that_is_not_the_sample(lab, tmp_path):
    db, _ = lab
    label_filings.main(["--db", str(db), "sample"])
    rows = _sheet(label_filings.SHEET)
    rows[0]["provider_ref"] = "0" * 64
    other = tmp_path / "other.csv"
    with other.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=label_filings.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    assert label_filings.main(["--db", str(db), "score", "--sheet", str(other)]) == 1


def test_wilson_interval():
    assert label_filings.wilson(0, 0) == (0.0, 1.0)
    lo, hi = label_filings.wilson(20, 20)
    assert hi == 1.0 and 0.83 < lo < 0.84          # 20/20 is not "100% accurate"
    lo, hi = label_filings.wilson(27, 30)
    assert lo < 0.9 < hi
