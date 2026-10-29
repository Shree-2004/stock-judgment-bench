"""Read-only view of the live filing store.

    streamlit run app/live_dashboard.py

Opens the database read-only and never fetches: the poller runs on its own
schedule, and a page someone happens to have open must not be what keeps data
flowing. Every timestamp shown says what it is — exchange time, first-seen
time, or the last successful poll — and the page states plainly how old its
newest data is rather than implying it is live.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live import store                                        # noqa: E402
from live.classify import CATEGORIES, CLASSIFIER_VERSION, classify  # noqa: E402
from live.filings import DEFAULT_SYMBOLS, IST, NSE_NAMES, tracked_symbol  # noqa: E402
from live.ingest import JOB_TYPE                              # noqa: E402

STALE_AFTER_MIN = 30      # poller runs every 15; two missed polls is a problem

st.set_page_config(page_title="Live filings", layout="wide")


def ist(ts: str | None) -> str:
    if not ts:
        return "—"
    return datetime.fromisoformat(ts).astimezone(IST).strftime("%d %b %Y %H:%M:%S IST")


@st.cache_data(ttl=60)
def load(db: str):
    conn = store.connect(db, read_only=True)
    filings = pd.read_sql_query(
        "SELECT company, subject, url, published_at, first_seen_at FROM filings "
        "ORDER BY published_at DESC", conn)
    runs = pd.read_sql_query("SELECT * FROM job_runs ORDER BY id DESC", conn)
    return filings, runs


st.title("Live filings")
st.caption("NSE corporate announcements, polled from the exchange's RSS feed. "
           "Research record only — not market data, not advice.")

if not store.DEFAULT_DB.exists():
    st.info("No store yet. Run `python -m scripts.live_ingest` first.")
    st.stop()

filings, runs = load(str(store.DEFAULT_DB))
now = datetime.now(timezone.utc)
ok_runs = runs[(runs.job_type == JOB_TYPE) & (runs.state == "ok")]

# --- freshness -------------------------------------------------------------
if ok_runs.empty:
    st.error("No successful poll yet.")
else:
    last_ok = ok_runs.iloc[0]
    age_min = (now - datetime.fromisoformat(last_ok.finished_at)).total_seconds() / 60
    msg = f"Last successful poll **{ist(last_ok.finished_at)}** — {age_min:.0f} min ago."
    (st.warning if age_min > STALE_AFTER_MIN else st.success)(
        msg + (" The poller may be stopped; data shown is not current." if age_min > STALE_AFTER_MIN else ""))
if not runs.empty and runs.iloc[0].state == "failed":
    st.error(f"Most recent poll FAILED: {runs.iloc[0].error}")

gaps = runs[runs.coverage_gap_s.notna() & (runs.coverage_gap_s > 0)]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filings stored", f"{len(filings):,}")
c2.metric("Polls", f"{len(runs):,}")
c3.metric("Failed polls", f"{(runs.state == 'failed').sum():,}")
c4.metric("Coverage gaps", f"{len(gaps):,}",
          help="Polls where the feed had fully turned over since the previous one, "
               "so filings in between may be missing from this record.")
if not gaps.empty:
    st.warning("Coverage gaps (filings published in these spans may be missing): " + "; ".join(
        f"{ist(r.feed_oldest)} minus {r.coverage_gap_s / 60:.0f} min" for r in gaps.itertuples()))

# --- tracked companies -----------------------------------------------------
filings["symbol"] = filings.company.map(tracked_symbol)
filings["category"] = [classify(s, u).category for s, u in zip(filings.subject, filings.url)]


def table(df: pd.DataFrame) -> None:
    show = pd.DataFrame({
        "published (exchange)": df.published_at.map(ist),
        "first seen (here)": df.first_seen_at.map(ist),
        "company": df.company,
        "category": df.category,
        "subject": df.subject,
        "source": df.url,
    })
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"source": st.column_config.LinkColumn(display_text="open")})


st.subheader("Tracked companies")
tabs = st.tabs([f"{s} ({(filings.symbol == s).sum()})" for s in DEFAULT_SYMBOLS])
for tab, sym in zip(tabs, DEFAULT_SYMBOLS):
    with tab:
        rows = filings[filings.symbol == sym]
        if rows.empty:
            st.write(f"No filings from *{NSE_NAMES[sym]}* since polling began.")
        else:
            table(rows)

st.subheader("All exchange filings")
fq, fc = st.columns([2, 1])
q = fq.text_input("Filter by company or subject")
cats = fc.multiselect("Category", list(CATEGORIES), default=[c for c in CATEGORIES if c != "fund_nav"],
                      help=f"Rule-based ({CLASSIFIER_VERSION}); see labels/README.md. "
                           "Fund NAVs are hidden by default.")
allf = filings[filings.category.isin(cats)]
if q:
    mask = allf.company.str.contains(q, case=False) | allf.subject.str.contains(q, case=False)
    allf = allf[mask]
table(allf.head(200))
st.caption(f"Showing {min(len(allf), 200)} of {len(allf):,}.")

# --- job log ---------------------------------------------------------------
with st.expander("Poll log (last 50)"):
    log = runs.head(50).copy()
    log["started"] = log.started_at.map(ist)
    st.dataframe(log[["id", "started", "state", "attempts", "fetched", "inserted",
                      "duplicates", "rejected", "coverage_gap_s", "error"]],
                 hide_index=True, width="stretch")
