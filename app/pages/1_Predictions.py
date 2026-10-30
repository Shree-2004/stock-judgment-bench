"""Read-only view of the M3 prediction ledger.

Opened from the sidebar of `streamlit run app/live_dashboard.py`.

Like the filings page, it never fetches and never writes. It recomputes the
hash chain on every load and checks the head against the witness file and
against what GitHub last showed, so the page says how much of the record can
be trusted rather than assuming it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live import store                                        # noqa: E402
from live.filings import IST                                  # noqa: E402
from live.ledger import CHAIN_FILE, WARMUP_END, verify_chain  # noqa: E402

st.set_page_config(page_title="Prediction ledger", layout="wide")

VERDICT_ORDER = {"BULLISH": 0, "NEUTRAL": 1, "BEARISH": 2}


def ist(ts: str | None) -> str:
    if not ts:
        return "—"
    return datetime.fromisoformat(ts).astimezone(IST).strftime("%d %b %Y %H:%M IST")


@st.cache_data(ttl=60)
def load(db: str):
    conn = store.connect(db, read_only=True)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "predictions" not in tables:
        return None, None, (True, 0, None), {}
    preds = pd.read_sql_query("SELECT * FROM predictions ORDER BY seq", conn)
    rounds = pd.read_sql_query("SELECT * FROM ledger_rounds ORDER BY id DESC", conn)
    urls = dict(conn.execute("SELECT provider_ref, url FROM filings WHERE url IS NOT NULL").fetchall())
    chain = verify_chain(store.connect(db, read_only=True))
    return preds, rounds, chain, urls


@st.cache_data(ttl=300)
def witness_heads() -> tuple[set[str], set[str] | None]:
    """Heads in the local chain.txt, and in origin/master's copy as of the last fetch."""
    local = set()
    if CHAIN_FILE.exists():
        local = {part[5:] for line in CHAIN_FILE.read_text(encoding="utf-8").splitlines()
                 for part in line.split() if part.startswith("head=")}
    rel = CHAIN_FILE.relative_to(ROOT).as_posix()
    res = subprocess.run(["git", "-C", str(ROOT), "show", f"origin/master:{rel}"],
                         capture_output=True, text=True)
    if res.returncode != 0:
        return local, None
    remote = {part[5:] for line in res.stdout.splitlines() for part in line.split() if part.startswith("head=")}
    return local, remote


st.title("Prediction ledger")
st.caption("One call per stock per NSE session from each model, made after the close and before the "
           "next open, from the frozen v1 prompt. Research record only — not advice.")

if not store.DEFAULT_DB.exists():
    st.info("No store yet.")
    st.stop()

preds, rounds, (chain_ok, n_checked, bad_seq), urls = load(str(store.DEFAULT_DB))
if preds is None or preds.empty:
    st.info("No predictions yet. The first round runs after today's close (`python -m scripts.live_predict`).")
    st.stop()

# --- integrity -------------------------------------------------------------
head = preds.iloc[-1].entry_hash
local, remote = witness_heads()
c1, c2, c3 = st.columns(3)
if chain_ok:
    c1.success(f"**Chain verified** — all {n_checked:,} rows recompute.")
else:
    c1.error(f"**Chain BROKEN** at seq {bad_seq}. A row was edited after it was written.")
if head in local:
    c2.success(f"**Head witnessed** in `ledger/chain.txt`: `{head[:12]}`")
else:
    c2.warning(f"Head `{head[:12]}` is not in `ledger/chain.txt` yet.")
if remote is None:
    c3.info("GitHub copy unknown (no `origin/master` fetched).")
elif head in remote:
    c3.success("**On GitHub** — this head is in the pushed history.")
else:
    c3.warning("**Not pushed yet** — the latest head is only on this machine.")

# --- session ---------------------------------------------------------------
# One track per model (README plan items 6-7), shown separately, never pooled.
models = list(dict.fromkeys(preds.model))
model = st.radio("Model", models, horizontal=True) if len(models) > 1 else models[0]

# A rerun appends attempt 2 for a stock whose attempt 1 failed; the call that
# counts is the latest attempt.
latest = (preds[preds.model == model].sort_values("attempt")
          .groupby(["session_date", "symbol"], as_index=False).last())
sessions = sorted(latest.session_date.unique(), reverse=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sessions", len(sessions))
m2.metric("Ledger rows", f"{len(preds):,}", help="Every attempt, including failures. Nothing is deleted.")
m3.metric("Calls made", f"{latest.verdict.notna().sum():,}")
m4.metric("Warm-up ends", WARMUP_END.strftime("%d %b %Y"),
          help="Before this date the filing store has less than the 21-day evidence window, "
               "so these sessions are excluded from the pre-registered analysis.")

session = st.selectbox("Session", sessions, format_func=lambda d: datetime.fromisoformat(d).strftime("%a %d %b %Y"))
day = latest[latest.session_date == session].copy()
day["n_evidence"] = day.evidence_json.map(lambda s: len(json.loads(s or "[]")))
day["order"] = day.verdict.map(VERDICT_ORDER).fillna(3)

if day.warmup.iloc[0]:
    st.info("Warm-up session: recorded and hash-chained, but excluded from the primary analysis.")

counts = Counter(day.verdict.fillna("failed"))
v1, v2, v3, v4 = st.columns(4)
v1.metric("Bullish", counts.get("BULLISH", 0))
v2.metric("Neutral", counts.get("NEUTRAL", 0))
v3.metric("Bearish", counts.get("BEARISH", 0))
v4.metric("Failed", counts.get("failed", 0))

show = day.sort_values(["order", "confidence", "symbol"], ascending=[True, False, True])
st.dataframe(pd.DataFrame({
    "symbol": show.symbol,
    "verdict": show.verdict.fillna("—"),
    "confidence": show.confidence,
    "close (₹)": show.price.round(2),
    "filings used": show.n_evidence,
    "made at": show.made_at.map(ist),
    "attempt": show.attempt,
    "error": show.error.fillna(""),
}), hide_index=True, width="stretch",
    column_config={"confidence": st.column_config.ProgressColumn(min_value=1, max_value=5, format="%d / 5")})

# --- one stock -------------------------------------------------------------
st.subheader("Why this call")
sym = st.selectbox("Stock", list(show.symbol))
row = show[show.symbol == sym].iloc[0]
brief = (json.loads(row.brief_json) if row.brief_json else {}).get("brief", {})
evidence = json.loads(row.evidence_json or "[]")

left, right = st.columns(2)
with left:
    st.markdown(f"**{row.verdict or 'no call'}**" + (f" · confidence {int(row.confidence)}/5" if row.confidence else ""))
    if brief.get("bull_case"):
        st.markdown(f"**Bull case.** {brief['bull_case']}")
    if brief.get("bear_case"):
        st.markdown(f"**Bear case.** {brief['bear_case']}")
    for d in brief.get("drivers", []):
        st.markdown(f"- {d.get('claim')}" + (f" → [{d['citation']}]" if d.get("citation") else ""))
    if row.error:
        st.error(row.error)
with right:
    st.markdown(f"**Filings the model was shown** ({len(evidence)})")
    if not evidence:
        st.write("None in the 21-day window. The call rests on price data alone.")
    for i, e in enumerate(evidence, 1):
        link = urls.get(e.get("provider_ref"))
        head_txt = e["headline"][:160] + ("…" if len(e["headline"]) > 160 else "")
        st.markdown(f"[{i}] `{e['category']}` {f'[{head_txt}]({link})' if link else head_txt}  \n"
                    f"<small>first seen here {ist(e['first_seen_at'])}</small>", unsafe_allow_html=True)

with st.expander("Raw model output and hashes"):
    st.code(row.raw_text or "(none)", language=None)
    st.caption(f"prev `{row.prev_hash}`  \nentry `{row.entry_hash}`  \n"
               f"model {row.model} · prompt {row.prompt_version} · classifier {row.classifier_version}")

with st.expander("Round log"):
    log = rounds.copy()
    log["started"] = log.started_at.map(ist)
    st.dataframe(log[["id", "session_date", "started", "state", "n_ok", "n_failed", "error"]],
                 hide_index=True, width="stretch")
