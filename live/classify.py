"""Rule-based filing categories (M2).

A filing gets exactly one category: *what new information does it carry for
someone holding the stock?* A newspaper reprint of results carries none (the
results were filed separately), so it is routine compliance, not results.

**How it decides.** NSE descriptions end in `|SUBJECT: <tag>`, and the tag is
chosen by the filer from the exchange's own list. For ~90% of filings the tag
alone settles it. A handful of tags say nothing — "Updates", "General
Updates", "Press Release", "Outcome of Board Meeting", "Others" — and for
those the free text is matched instead, with a per-tag fallback when nothing
in the text is specific.

Every result carries the id of the rule that fired, so a wrong label points
at the line that produced it.

**Why it is computed on read, not stored.** `filings` is append-only. Labels
are a derived view, and a rule fix must not require rewriting history.
`CLASSIFIER_VERSION` names the rule set, so anything that *does* persist a
label (the M3 ledger) can record which rules produced it.

**Frozen before it was scored.** The rules were written from the tag counts of
the first 1,460 filings and a sample of ~80 descriptions. The hand-label
sample is drawn only from filings first seen after `RULES_FROZEN_AT`, so the
classifier is scored on text it was not written against.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

CLASSIFIER_VERSION = "rules-v1"
RULES_FROZEN_AT = "2026-10-29T06:30:00+00:00"      # 12:00 IST

CATEGORIES: dict[str, str] = {
    "results":            "financial results, results press releases, results updates",
    "board_meeting":      "notice of an upcoming board meeting, or an outcome that names no decision",
    "investor_comms":     "analyst/investor meets, call transcripts, investor presentations, KPIs",
    "corporate_action":   "dividend, record date, bonus, split, buyback, rights issue, name change",
    "capital_raise":      "allotment, issue of securities, fund raising, ESOP grants/exercise",
    "deal":               "acquisition, disposal, merger, demerger, scheme, open offer, JV, material agreement",
    "business_update":    "orders and contracts, capacity, launches, operations, other company news",
    "management_change":  "director/KMP appointment, resignation, cessation, retirement, demise",
    "credit_rating":      "new, revised or reaffirmed credit ratings",
    "legal_regulatory":   "regulatory action, fines, litigation, fraud, default",
    "debt_and_proceeds":  "debt servicing, security cover, guarantees, use of issue proceeds",
    "shareholder_meeting": "AGM/EGM/postal ballot notices, proceedings, voting results",
    "routine_compliance": "newspaper reprints, trading window, insider-code and certificate filings",
    "fund_nav":           "mutual fund / ETF NAV declarations (not company filings)",
    "other":              "nothing above applies, or the filing says too little to tell",
}

# Tags that carry no information by themselves -> read the text instead.
# Value: the category to fall back to when the text matches nothing.
GENERIC_TAGS: dict[str, str] = {
    "updates": "business_update",
    "general updates": "business_update",
    "material events": "business_update",
    "others": "other",
    "press release": "business_update",
    "press release (revised)": "business_update",
    "outcome of board meeting": "board_meeting",
    "": "other",
}

# (rule id, category, pattern) — matched against the normalised tag.
# Order matters only where tags could overlap; first match wins.
TAG_RULES: list[tuple[str, str, str]] = [
    ("tag.nav",         "fund_nav",            r"declaration of nav"),
    ("tag.results",     "results",             r"financial results|integrated filing- financial|limited review"),
    ("tag.bm_notice",   "board_meeting",       r"board meeting intimation"),
    ("tag.investor",    "investor_comms",      r"analyst|investor meet|con\. call|investor presentation"),
    ("tag.open_offer",  "deal",                r"open offer"),
    ("tag.corp_action", "corporate_action",    r"dividend|record date|rights issue|buy ?back|bonus|split|name (and symbol )?change"),
    ("tag.capital",     "capital_raise",       r"allotment|issue of securities|alteration of capital|fund raising|esop|esos|esps|options to purchase"),
    ("tag.deal",        "deal",                r"acquisition|sale or disposal|amalgamation|merger|scheme of arrangement|agreements|diversification|disinvestment"
                                               r"|restructuring|tie up"),
    ("tag.business",    "business_update",     r"bagging|awarding of order|capacity addition|product launch|commencement of commercial|new line"),
    ("tag.management",  "management_change",   r"director|resignation|appointment|change in management|cessation|retirement|demise"),
    ("tag.rating",      "credit_rating",       r"credit rating"),
    ("tag.legal",       "legal_regulatory",    r"action\(s\) taken|actions initiated|orders passed|litigation|fraud|default"),
    ("tag.debt",        "debt_and_proceeds",   r"redemption|payment of interest|security cover|regulation 5[0-2]|deviation|monitoring agency|utili[sz]ation|guarantee"),
    ("tag.agm",         "shareholder_meeting", r"shareholders meeting|notice of shareholders"),
    ("tag.compliance",  "routine_compliance",  r"newspaper|trading window|certificate under|compliance report|regulation (6|7|13|23)\b"
                                               r"|integrated filing- governance|annual disclosure|committee meeting|aoa|moa"
                                               r"|valuation report|noc|price movement|reasons for delayed|related party"),
]

# (rule id, category, pattern) — matched against the lower-cased free text,
# only for generic tags. First match wins, so specific beats vague: a press
# release about results is results before it is business news.
TEXT_RULES: list[tuple[str, str, str]] = [
    ("text.investor",    "investor_comms",      r"investor presentation|analyst|investor meet|con(ference)? call|earnings call|transcript|key performance indicator|\bkpis?\b"),
    ("text.results",     "results",             r"financial results|\bresults\b|performance update|\brevenue\b|\bebitda\b|net profit|\bprofit\b"
                                               r"|total income|financial (highlights|performance|information)|\bq[1-4]\s*(&\s*h[12]\s*)?fy\s*'?\d{2}"
                                               r"|press release (for|on) (the )?(quarter|half|q[1-4])"),
    ("text.rating",      "credit_rating",       r"credit rating"),
    ("text.legal",       "legal_regulatory",    r"\bfine\b|penalt|litigation|show cause|tribunal|\bnclt\b|\bcourt\b|\bfraud\b|\bdefault\b|cautionary"),
    ("text.deal",        "deal",                r"acqui|merger|amalgamation|demerger|scheme of arrangement|incorporation of .*subsidiary|joint venture|divest|open offer"
                                               r"|investment in"),
    ("text.business",    "business_update",     r"\border\b.*\b(worth|from|for)\b|secur(es|ed|ing) .*order|contract|letter of award|capacity|commission|launch|deployment|on-?hire|exports?\b|production"),
    ("text.corp_action", "corporate_action",    r"dividend|record date|bonus|sub-?division|split|buy-?back|rights issue"),
    ("text.capital",     "capital_raise",       r"allotment|esop|esos|exercise of options|options granted|fund rais|raising of .*funds|preferential|\bqip\b|warrants"),
    ("text.debt",        "debt_and_proceeds",   r"deviation|variation|monitoring agency|utili[sz]ation|\bncds?\b|debenture|interest and principal"
                                               r"|payment of interest|security cover"),
    ("text.agm",         "shareholder_meeting", r"general meeting|\bagm\b|\begm\b|postal ballot|e-?voting|scrutini"),
    ("text.compliance",  "routine_compliance",  r"newspaper|trading window|regulation 8\(2\)|\bupsi\b|insider|\bpit\b|compliance|certificate|regulation 30\(5\)"
                                               r"|regulation 6\(1\)|fair disclosure|committee|movement in price|website|corporate identification|\bcin\b"),
    ("text.bm_notice",   "board_meeting",       r"intimation of (the )?board meeting|meeting of (the )?board of directors|board meeting (scheduled|to be held)"),
    ("text.management",  "management_change",   r"appoint|resign|cessation|\bdirectors?\b|\bkmp\b|key managerial|\bceo\b|\bcfo\b|retire"
                                               r"|chief \w+ officer|senior management|designat"),
]

_TAG_RULES = [(rid, cat, re.compile(p)) for rid, cat, p in TAG_RULES]
_TEXT_RULES = [(rid, cat, re.compile(p)) for rid, cat, p in TEXT_RULES]

assert {c for _, c, _ in TAG_RULES + TEXT_RULES} | set(GENERIC_TAGS.values()) <= set(CATEGORIES)


@dataclass(frozen=True)
class Classification:
    category: str
    rule: str          # which rule fired, e.g. "tag.results" or "fallback.press release"


def split_subject(subject: str) -> tuple[str, str]:
    """`"<text> |SUBJECT: <tag>"` -> (text, normalised tag). Tag is "" if absent."""
    text, sep, tag = (subject or "").rpartition("|SUBJECT:")
    if not sep:
        return subject or "", ""
    return text.strip(), normalise_tag(tag)


def normalise_tag(tag: str) -> str:
    """Drop the variants NSE appends to the same tag: "-XBRL", "(Sub-para 4-Para B)",
    "Update-" prefixes, doubled spaces."""
    tag = tag.strip().lower()
    tag = re.sub(r"^update-", "", tag)
    tag = re.sub(r"-xbrl$", "", tag)
    tag = re.sub(r"\s*-?\(sub-para[^)]*\)", "", tag)
    return " ".join(tag.split())


def classify(subject: str) -> Classification:
    text, tag = split_subject(subject)

    if tag not in GENERIC_TAGS:
        for rid, cat, pat in _TAG_RULES:
            if pat.search(tag):
                return Classification(cat, rid)
        return Classification("other", "unmatched_tag")

    low = text.lower()
    for rid, cat, pat in _TEXT_RULES:
        if pat.search(low):
            return Classification(cat, rid)
    if says_little(low):
        return Classification("other", "empty")
    return Classification(GENERIC_TAGS[tag], f"fallback.{tag or 'no tag'}")


_BOILERPLATE = re.compile(
    r"^.*?has (informed|submitted to) the exchange,? (about|regarding|under)?"
    r"( a( revised)? press release dated [^,|]*(, titled)?)?")
_FILLER = set("""a an the of on in for to and by dated held regarding about intimation
    press release revised general update updates outcome board meeting others
    please find enclosed herewith attached copy being issued company
    regulation regulations sebi lodr listing obligations disclosure requirements
    january february march april may june july august september october november december""".split())


def says_little(low_text: str) -> bool:
    """True when the text is boilerplate plus the tag restated, e.g. "X has informed
    the Exchange about General Updates" or a press release titled "Press Release"."""
    rest = _BOILERPLATE.sub("", low_text)
    words = [w for w in re.findall(r"[a-z]+", rest) if w not in _FILLER]
    return len(words) < 2


def rules_fingerprint() -> str:
    """Hash of everything that decides a label. The label sample records it, so
    a rule edit made after labelling is detected even if nobody bumps the version."""
    blob = json.dumps([TAG_RULES, TEXT_RULES, GENERIC_TAGS, sorted(_FILLER), _BOILERPLATE.pattern])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
