"""Export the live ledger and daily prices as static JSON for `site/`.

    python -m scripts.export_site               # writes site/data/*.json
    python -m http.server 8700 -d site          # then open http://localhost:8700

The site is static so it can be served from anywhere, GitHub Pages included,
with no server that could write to the ledger. This script is the only bridge:
it opens the store read-only, recomputes the hash chain, and fetches daily
candles. It never writes to the store.

Prices are raw (unadjusted) daily OHLC, the same basis as the `price` recorded
on each ledger row, so a call's marker sits on the close it read.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, time, timezone
from pathlib import Path

from live import store
from live.filings import IST, NSE_NAMES
from live.ledger import CHAIN_FILE, ROOT, WARMUP_END, verify_chain

OUT = ROOT / "site" / "data"
PRICE_PERIOD = "1y"
CLOSE_SETTLED = time(16, 0)       # NSE closes 15:30 IST; give the daily bar time to settle


def _heads(text: str) -> set[str]:
    return {part[5:] for line in text.splitlines() for part in line.split() if part.startswith("head=")}


def witness(head: str) -> dict:
    """Is `head` in the local chain.txt, and in origin/master's copy as of the last fetch?"""
    local = _heads(CHAIN_FILE.read_text(encoding="utf-8")) if CHAIN_FILE.exists() else set()
    rel = CHAIN_FILE.relative_to(ROOT).as_posix()
    res = subprocess.run(["git", "-C", str(ROOT), "show", f"origin/master:{rel}"], capture_output=True, text=True)
    return {"witnessed": head in local, "pushed": None if res.returncode else head in _heads(res.stdout)}


def export_ledger(conn) -> dict:
    conn.row_factory = None
    cols = [d[1] for d in conn.execute("PRAGMA table_info(predictions)")]
    rows = [dict(zip(cols, r)) for r in conn.execute("SELECT * FROM predictions ORDER BY seq")]
    urls = dict(conn.execute("SELECT provider_ref, url FROM filings WHERE url IS NOT NULL").fetchall())

    # A rerun appends attempt 2 for a stock whose attempt 1 failed; the call
    # that counts is the latest attempt.
    latest: dict[tuple[str, str], dict] = {}
    for r in rows:
        latest[(r["session_date"], r["symbol"])] = r

    sessions: dict[str, dict] = {}
    for (session, sym), r in sorted(latest.items()):
        brief = (json.loads(r["brief_json"]) if r["brief_json"] else {}).get("brief", {})
        evidence = json.loads(r["evidence_json"] or "[]")
        for e in evidence:
            e["url"] = urls.get(e.get("provider_ref"))
        sessions.setdefault(session, {})[sym] = {
            "verdict": r["verdict"], "confidence": r["confidence"], "price": r["price"],
            "made_at": r["made_at"], "attempt": r["attempt"], "warmup": bool(r["warmup"]), "error": r["error"],
            "bull_case": brief.get("bull_case"), "bear_case": brief.get("bear_case"),
            "drivers": brief.get("drivers", []), "evidence": evidence,
            "entry_hash": r["entry_hash"],
        }

    ok, n, bad = verify_chain(conn)
    head = rows[-1]["entry_hash"] if rows else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": rows[-1]["model"] if rows else None,
        "prompt_version": rows[-1]["prompt_version"] if rows else None,
        "warmup_end": WARMUP_END.isoformat(),
        "chain": {"ok": ok, "rows": n, "bad_seq": bad, "head": head, **(witness(head) if head else {})},
        "names": NSE_NAMES,
        "sessions": sessions,
    }


def export_prices(symbols) -> dict:
    import yfinance as yf
    tickers = [f"{s}.NS" for s in symbols]
    df = yf.download(tickers, period=PRICE_PERIOD, interval="1d", auto_adjust=False, group_by="ticker",
                     progress=False, threads=True)
    # During market hours yfinance returns today's bar half-formed; showing it
    # would put a moving candle next to calls that read only finished closes.
    now = datetime.now(IST)
    unfinished = now.date().isoformat() if now.time() < CLOSE_SETTLED else None
    out = {}
    for sym, tk in zip(symbols, tickers):
        if tk not in df.columns.get_level_values(0):
            continue
        d = df[tk].dropna(subset=["Open", "High", "Low", "Close"])
        d = d[d.index.strftime("%Y-%m-%d") != unfinished]
        out[sym] = [{"time": idx.strftime("%Y-%m-%d"), "open": round(float(r.Open), 2),
                     "high": round(float(r.High), 2), "low": round(float(r.Low), 2),
                     "close": round(float(r.Close), 2), "volume": int(r.Volume or 0)}
                    for idx, r in d.iterrows()]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    ap.add_argument("--no-prices", action="store_true", help="refresh the ledger only")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    ledger = export_ledger(store.connect(args.db, read_only=True))
    (OUT / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
    print(f"ledger: {sum(len(v) for v in ledger['sessions'].values())} calls over {len(ledger['sessions'])} "
          f"session(s); chain {'OK' if ledger['chain']['ok'] else 'BROKEN'}")
    if not args.no_prices:
        prices = export_prices(list(NSE_NAMES))
        (OUT / "prices.json").write_text(json.dumps(prices), encoding="utf-8")
        missing = sorted(set(NSE_NAMES) - set(prices))
        print(f"prices: {len(prices)} symbols" + (f"; missing {missing}" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
