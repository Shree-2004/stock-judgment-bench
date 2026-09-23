"""Live view of the benchmark. Read-only, reads the same frozen artefacts the
report does.

    streamlit run app/dashboard.py

This is a monitoring surface, not the product. The plan cut a Streamlit *stock
app* as noise, and that still stands — nothing here generates a brief, changes
a score, or touches the network. It reads `cache/` and renders it.

Two rules it inherits from the CLI report, because they are the whole point:

- **Partial data is labelled partial.** The first baseline run printed a tidy
  table built on half a grid; a dashboard that quietly renders 300 of 3,690
  briefs as a finished result would repeat that mistake with better styling.
- **Nothing that cannot be measured gets a number.** `always_bullish` shows a
  dash, not a zero, because under cross-sectional scoring it is genuinely
  unscoreable rather than merely bad.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings                                   # noqa: E402
from data.universe import UNIVERSE                            # noqa: E402
from data.windows import build_windows                        # noqa: E402
from evals.baselines import BASELINES, build_baseline_rows    # noqa: E402
from evals.grid import grid_stats, load_grid                  # noqa: E402
from evals.scoring import VERDICT_SCORE, score_l3             # noqa: E402

BRIEFS = ROOT / "cache" / "briefs"
REFRESH_S = 20

st.set_page_config(page_title="stock-judgment-bench", layout="wide")


def slug(model: str, effort: str) -> str:
    s = model.replace("/", "_").replace(":", "_")
    return f"{s}@{effort}" if effort and "gpt-oss" in model else s


@st.cache_data(ttl=REFRESH_S, show_spinner=False)
def scan_briefs(namespace: str, prompt_version: str) -> dict:
    """One pass over the brief cache. Cheap enough to redo every refresh."""
    root = BRIEFS / namespace / prompt_version
    out = {
        "ok": 0, "failed": 0, "unparsed": 0,
        "l1_scores": [], "clean": 0, "wrong": 0, "unverifiable": 0,
        "verdicts": Counter(), "confidence": Counter(), "tokens": 0,
        "rows": {}, "latest": None, "errors": Counter(), "by_ticker": {},
    }
    if not root.exists():
        return out

    for p in root.rglob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        gen = d.get("generated_at")
        if gen and (out["latest"] is None or gen > out["latest"]):
            out["latest"] = gen
        if d.get("error"):
            out["failed"] += 1
            out["errors"][str(d["error"]).split(":")[0]] += 1
            continue
        out["ok"] += 1
        brief = d.get("brief") or {}
        l1 = d.get("l1") or {}
        if not brief.get("parsed"):
            out["unparsed"] += 1
        v = brief.get("verdict")
        if v:
            tk = d.get("ticker", "")
            out["verdicts"][v] += 1
            out["rows"].setdefault(d.get("cutoff", ""), []).append(
                (VERDICT_SCORE.get(v, 0.0), tk, v)
            )
            out["by_ticker"].setdefault(tk, []).append({
                "cutoff": d.get("cutoff", ""),
                "verdict": v,
                "confidence": brief.get("confidence"),
                "l1": l1.get("score"),
                "clean": bool(l1.get("clean")),
            })
        if brief.get("confidence") is not None:
            out["confidence"][brief["confidence"]] += 1
        if l1.get("score") is not None:
            out["l1_scores"].append(float(l1["score"]))
        out["clean"] += int(bool(l1.get("clean")))
        out["wrong"] += int(l1.get("wrong") or 0)
        out["unverifiable"] += int(l1.get("unverifiable") or 0)
        out["tokens"] += (d.get("usage") or {}).get("total_tokens", 0) or 0
    return out


@st.cache_data(ttl=300, show_spinner=False)
def load_frozen():
    windows = build_windows()
    tickers = [c.ticker for c in UNIVERSE]
    cutoffs = [w.cutoff for w in windows]
    stats = grid_stats(tickers, cutoffs)
    grid = load_grid(tickers, cutoffs, allow_incomplete=True)
    baselines = {
        b.name: score_l3(b.name, build_baseline_rows(grid)[b.name]).summary()
        for b in BASELINES
    }
    return len(windows), len(tickers), stats, grid, baselines


def fmt(v, pct=False, places=4):
    if v is None:
        return "—"
    return f"{v:.1%}" if pct else f"{v:.{places}f}"


# ---------------------------------------------------------------------------

st.title("stock-judgment-bench")
st.caption(
    "Does an LLM's stock judgement contain anything, once lookahead leakage "
    "and market drift are removed? Expected answer: no."
)

n_windows, n_tickers, gstats, grid, baselines = load_frozen()
namespace = slug(settings.llm_model, settings.reasoning_effort)
b = scan_briefs(namespace, "v1_structured")

total = n_tickers * n_windows
done = b["ok"] + b["failed"]

st.sidebar.header("run")
st.sidebar.write(f"**model** `{settings.llm_model}`")
st.sidebar.write(f"**namespace** `{namespace}`")
st.sidebar.write(f"**universe** {n_tickers} tickers x {n_windows} dates")
st.sidebar.write(f"**last brief** {b['latest'] or '—'}")
st.sidebar.write(f"**tokens used** {b['tokens']:,}")
if st.sidebar.button("refresh now"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"auto-refreshes every {REFRESH_S}s")

complete = done >= total and b["failed"] == 0
if not complete:
    st.warning(
        f"**PARTIAL — {done:,} of {total:,} briefs ({done/total:.1%}). "
        f"Not a result yet.** Numbers below will move.",
        icon="⏳",
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("briefs generated", f"{b['ok']:,}", f"{b['failed']} failed" if b["failed"] else "no failures")
c2.metric("grid cells", f"{gstats['ok']:,}/{gstats['expected']:,}",
          "complete" if gstats["complete"] else "incomplete")
mean_l1 = sum(b["l1_scores"]) / len(b["l1_scores"]) if b["l1_scores"] else None
c3.metric("L1 grounding", fmt(mean_l1, pct=True),
          f"{b['wrong']} wrong figures")
c4.metric("L1 clean rate", fmt(b["clean"] / b["ok"], pct=True) if b["ok"] else "—",
          f"{b['unverifiable']} invented labels")

st.progress(min(1.0, done / total) if total else 0.0)

left, right = st.columns([1, 1])

with left:
    st.subheader("verdict distribution")
    if b["verdicts"]:
        order = ["BULLISH", "NEUTRAL", "BEARISH"]
        counts = {k: b["verdicts"].get(k, 0) for k in order}
        st.bar_chart(counts, height=220)
        tot = sum(counts.values())
        top = max(counts, key=counts.get)
        if tot and counts[top] / tot > 0.7:
            st.warning(
                f"**{counts[top]/tot:.0%} {top}.** A near-constant verdict is "
                f"structurally unscoreable — the same reason `always_bullish` "
                f"reports nothing. L3 may have little to measure.",
                icon="⚠️",
            )
    else:
        st.info("no verdicts yet")

    if b["confidence"]:
        st.caption("confidence values used: " +
                   ", ".join(f"{k}×{v}" for k, v in sorted(b["confidence"].items())))

with right:
    st.subheader("L3 — outcome, one observation per date")
    returns = {(r["cutoff"], r["ticker"]): r["forward_return"]
               for rows in grid.values() for r in rows}
    model_rows: dict = {}
    for cutoff, entries in b["rows"].items():
        for score, ticker, verdict in entries:
            fwd = returns.get((cutoff, ticker))
            if fwd is not None:
                model_rows.setdefault(cutoff, []).append((score, fwd, verdict))

    rows = [("model", score_l3("model", model_rows).summary())]
    rows += [(n, s) for n, s in baselines.items()]
    st.dataframe(
        [{"strategy": n,
          "dates": s["dates"],
          "rank corr": fmt(s["mean_rank_correlation"]),
          "t": fmt(s["rank_correlation_t"], places=2),
          "spread %": fmt(s["mean_spread_pct"], places=3),
          "hit": fmt(s["spread_hit_rate"], places=3)} for n, s in rows],
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "`always_bullish` shows dashes because cross-sectional scoring makes "
        "market drift unexploitable — it is unscoreable, not merely bad. "
        "Pre-registered prediction: if the model re-reads momentum it should "
        "land near momentum's rank correlation, not near zero."
    )

st.divider()
st.subheader("per-stock breakdown")

st.info(
    "**Per-stock numbers are diagnostics, not the benchmark result.** The "
    "benchmark ranks the 41 names *against each other on the same date*, which "
    "is what removes the market. A single stock has no cross-section, so the "
    "best available per-stock measure is whether the verdict matched that "
    "stock's return **relative to the Nifty** — market-neutral per observation, "
    "but far weaker than the real metric and not what any conclusion rests on.",
    icon="ℹ️",
)

excess = {(r["cutoff"], r["ticker"]): r.get("excess_return")
          for rows in grid.values() for r in rows}

def directional_hit(verdict: str, exc: float | None) -> bool | None:
    """NEUTRAL makes no directional claim, so it is excluded rather than
    scored as a miss — counting silence as a wrong answer would punish the
    model for declining to guess."""
    if exc is None or verdict == "NEUTRAL":
        return None
    return (exc > 0) if verdict == "BULLISH" else (exc < 0)

table = []
for tk, entries in sorted(b["by_ticker"].items()):
    n = len(entries)
    counts = Counter(e["verdict"] for e in entries)
    hits = [directional_hit(e["verdict"], excess.get((e["cutoff"], tk))) for e in entries]
    called = [h for h in hits if h is not None]
    l1s = [e["l1"] for e in entries if e["l1"] is not None]
    exc_vals = [excess.get((e["cutoff"], tk)) for e in entries]
    exc_vals = [v for v in exc_vals if v is not None]
    table.append({
        "ticker": tk.replace(".NS", ""),
        "briefs": n,
        "bull": counts.get("BULLISH", 0),
        "neut": counts.get("NEUTRAL", 0),
        "bear": counts.get("BEARISH", 0),
        "L1": fmt(sum(l1s) / len(l1s), pct=True) if l1s else "—",
        "directional calls": len(called),
        "hit vs Nifty": fmt(sum(called) / len(called), pct=True) if called else "—",
        "mean excess %": fmt(sum(exc_vals) / len(exc_vals), places=2) if exc_vals else "—",
    })

if table:
    st.dataframe(table, use_container_width=True, hide_index=True, height=420)
    st.caption(
        "`directional calls` excludes NEUTRAL verdicts. A hit rate near 50% is "
        "the null. With ~90 windows per stock, a single ticker cannot resolve "
        "anything on its own — read the column for outliers worth inspecting, "
        "not as evidence."
    )

    pick = st.selectbox("inspect one stock's history",
                        sorted(t["ticker"] for t in table))
    entries = sorted(b["by_ticker"].get(f"{pick}.NS", []), key=lambda e: e["cutoff"])
    st.dataframe(
        [{"date": e["cutoff"],
          "verdict": e["verdict"],
          "conf": e["confidence"],
          "L1": fmt(e["l1"], pct=True) if e["l1"] is not None else "—",
          "excess vs Nifty %": fmt(excess.get((e["cutoff"], f"{pick}.NS")), places=2),
          "hit": {True: "yes", False: "no", None: "—"}[
              directional_hit(e["verdict"], excess.get((e["cutoff"], f"{pick}.NS")))]}
         for e in entries],
        use_container_width=True, hide_index=True, height=340,
    )
else:
    st.info("no per-stock data yet")

if b["errors"]:
    st.subheader("failures by type")
    st.write(dict(b["errors"]))

st.caption(
    f"read-only · refreshed {datetime.now():%H:%M:%S} · "
    f"L2 tier-2 (semantic overlap) is deliberately not shown: measurement "
    f"proved it flags correct paraphrases, so it is not a metric."
)

st.markdown(
    f"<meta http-equiv='refresh' content='{REFRESH_S}'>", unsafe_allow_html=True
)
