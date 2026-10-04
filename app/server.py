"""Tiny stdlib server behind the live dashboard.

Streamlit re-runs a whole Python script per interaction, which makes anything
beyond a form feel sluggish and constrains the layout to its widget set. This
does the one thing actually needed: compute the benchmark state from `cache/`
and hand it over as JSON. The page is plain HTML/JS and owns its own design.

Read-only by construction. No route writes anything, touches the network, or
generates a brief — it reads frozen artefacts, same as `scripts/report.py`.

    python -m app.server            # http://localhost:8600
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

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
STATIC = Path(__file__).resolve().parent / "static"

_cache: dict = {"at": 0.0, "state": None}
_lock = threading.Lock()
CACHE_TTL_S = 8.0

# The grid and baselines are frozen; computing them once at boot keeps each
# poll cheap enough to run every few seconds.
_frozen: dict = {}


def _slug(model: str, effort: str) -> str:
    s = model.replace("/", "_").replace(":", "_")
    return f"{s}@{effort}" if effort and "gpt-oss" in model else s


def _load_frozen() -> dict:
    windows = build_windows()
    tickers = [c.ticker for c in UNIVERSE]
    cutoffs = [w.cutoff for w in windows]
    grid = load_grid(tickers, cutoffs, allow_incomplete=True)
    rows = build_baseline_rows(grid)
    return {
        "n_windows": len(windows),
        "n_tickers": len(tickers),
        "grid_stats": grid_stats(tickers, cutoffs),
        "returns": {f"{r['cutoff']}|{r['ticker']}": r["forward_return"]
                    for rs in grid.values() for r in rs},
        "excess": {f"{r['cutoff']}|{r['ticker']}": r.get("excess_return")
                   for rs in grid.values() for r in rs},
        "price": {f"{r['cutoff']}|{r['ticker']}": r.get("price")
                  for rs in grid.values() for r in rs},
        "mom": {f"{r['cutoff']}|{r['ticker']}": r.get("change_pct_1mo")
                for rs in grid.values() for r in rs},
        "baselines": [dict(score_l3(b.name, rows[b.name]).summary(),
                           describe=b.describe) for b in BASELINES],
    }


def _scan(namespace: str, prompt_version: str) -> dict:
    root = BRIEFS / namespace / prompt_version
    ok = failed = unparsed = clean = wrong = unverifiable = tokens = 0
    l1: list[float] = []
    verdicts: Counter = Counter()
    confidence: Counter = Counter()
    by_date: dict[str, list] = {}
    by_ticker: dict[str, list] = {}
    errors: Counter = Counter()
    latest = None

    if root.exists():
        for p in root.rglob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            gen = d.get("generated_at")
            if gen and (latest is None or gen > latest):
                latest = gen
            if d.get("error"):
                failed += 1
                errors[str(d["error"]).split(":")[0]] += 1
                continue
            ok += 1
            brief = d.get("brief") or {}
            l1d = d.get("l1") or {}
            if not brief.get("parsed"):
                unparsed += 1
            if l1d.get("score") is not None:
                l1.append(float(l1d["score"]))
            clean += int(bool(l1d.get("clean")))
            wrong += int(l1d.get("wrong") or 0)
            unverifiable += int(l1d.get("unverifiable") or 0)
            tokens += (d.get("usage") or {}).get("total_tokens", 0) or 0
            if brief.get("confidence") is not None:
                confidence[brief["confidence"]] += 1
            v = brief.get("verdict")
            if v:
                tk, cut = d.get("ticker", ""), d.get("cutoff", "")
                verdicts[v] += 1
                by_date.setdefault(cut, []).append((VERDICT_SCORE.get(v, 0.0), tk, v))
                by_ticker.setdefault(tk, []).append({
                    "cutoff": cut, "verdict": v,
                    "confidence": brief.get("confidence"),
                    "l1": l1d.get("score"),
                })

    return {
        "ok": ok, "failed": failed, "unparsed": unparsed,
        "mean_l1": (sum(l1) / len(l1)) if l1 else None,
        "clean_rate": (clean / ok) if ok else None,
        "wrong": wrong, "unverifiable": unverifiable, "tokens": tokens,
        "verdicts": dict(verdicts), "confidence": dict(sorted(confidence.items())),
        "errors": dict(errors), "latest": latest,
        "_by_date": by_date, "_by_ticker": by_ticker,
    }


def _hit(verdict: str, excess: float | None):
    """NEUTRAL makes no directional claim, so it is excluded rather than scored
    as a miss — counting silence as a wrong answer would punish the model for
    declining to guess."""
    if excess is None or verdict == "NEUTRAL":
        return None
    return (excess > 0) if verdict == "BULLISH" else (excess < 0)


def build_state() -> dict:
    f = _frozen
    ns = _slug(settings.llm_model, settings.reasoning_effort)
    s = _scan(ns, "v1_structured")

    model_rows: dict = {}
    for cutoff, entries in s["_by_date"].items():
        for score, ticker, verdict in entries:
            fwd = f["returns"].get(f"{cutoff}|{ticker}")
            if fwd is not None:
                model_rows.setdefault(cutoff, []).append((score, fwd, verdict))
    model_l3 = dict(score_l3("model", model_rows).summary(), describe="the model under test")

    per_stock = []
    for tk, entries in sorted(s["_by_ticker"].items()):
        counts = Counter(e["verdict"] for e in entries)
        hits, excs = [], []
        for e in entries:
            key = f"{e['cutoff']}|{tk}"
            ex = f["excess"].get(key)
            h = _hit(e["verdict"], ex)
            if h is not None:
                hits.append(h)
            if ex is not None:
                excs.append(ex)
        l1s = [e["l1"] for e in entries if e["l1"] is not None]
        per_stock.append({
            "ticker": tk.replace(".NS", ""),
            "briefs": len(entries),
            "bull": counts.get("BULLISH", 0),
            "neut": counts.get("NEUTRAL", 0),
            "bear": counts.get("BEARISH", 0),
            "l1": (sum(l1s) / len(l1s)) if l1s else None,
            "calls": len(hits),
            "hit": (sum(hits) / len(hits)) if hits else None,
            "mean_excess": (sum(excs) / len(excs)) if excs else None,
            "history": [
                {"cutoff": e["cutoff"], "verdict": e["verdict"],
                 "confidence": e["confidence"], "l1": e["l1"],
                 "price": f["price"].get(f"{e['cutoff']}|{tk}"),
                 "mom": f["mom"].get(f"{e['cutoff']}|{tk}"),
                 "fwd": f["returns"].get(f"{e['cutoff']}|{tk}"),
                 "excess": f["excess"].get(f"{e['cutoff']}|{tk}"),
                 "hit": _hit(e["verdict"], f["excess"].get(f"{e['cutoff']}|{tk}"))}
                for e in sorted(entries, key=lambda x: x["cutoff"])
            ],
        })

    total = f["n_tickers"] * f["n_windows"]
    done = s["ok"] + s["failed"]
    return {
        "model": settings.llm_model,
        "namespace": ns,
        "n_tickers": f["n_tickers"],
        "n_windows": f["n_windows"],
        "total": total,
        "done": done,
        "complete": done >= total and s["failed"] == 0,
        "grid": f["grid_stats"],
        "progress": {k: s[k] for k in
                     ("ok", "failed", "unparsed", "mean_l1", "clean_rate",
                      "wrong", "unverifiable", "tokens", "verdicts",
                      "confidence", "errors", "latest")},
        "l3": [model_l3] + f["baselines"],
        "per_stock": per_stock,
        "generated_at": time.strftime("%H:%M:%S"),
    }


def cached_state() -> dict:
    with _lock:
        if _cache["state"] is not None and time.time() - _cache["at"] < CACHE_TTL_S:
            return _cache["state"]
    state = build_state()
    with _lock:
        _cache.update(at=time.time(), state=state)
    return state


def window_detail(ticker: str, cutoff: str) -> dict:
    """Everything one brief was built from, and what it said.

    This is the archived evidence for a single (ticker, date) — the filings
    the model was actually shown on that day, not a live feed. A live news
    feed would put today's information beside a 2021 verdict, which is the
    exact lookahead leak the whole benchmark is built to prevent.
    """
    from datetime import date as _date

    from data.news_data import get_disclosures
    from scripts.run_briefs import model_slug, result_path

    symbol = ticker.replace(".NS", "")
    path = result_path("v1_structured", settings.llm_model, symbol, cutoff)
    if not path.exists():
        return {"error": f"no brief for {symbol} @ {cutoff}"}
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("error"):
        return {"error": rec["error"], "ticker": ticker, "cutoff": cutoff}

    brief = rec.get("brief") or {}
    try:
        bundle = get_disclosures(symbol, _date.fromisoformat(cutoff), limit=5)
        filings = [
            {"n": i, "filed_at": (it.filed_at or "")[:10],
             "category": it.category, "headline": it.headline}
            for i, it in enumerate(bundle.items, 1)
        ]
        considered, noise = bundle.considered, bundle.dropped_as_noise
    except Exception as exc:                                   # noqa: BLE001
        filings, considered, noise = [], 0, 0
        brief.setdefault("_filings_error", str(exc))

    # Resolve each stated reason to the filing it cited, so a fabricated or
    # out-of-range citation is visible rather than implied.
    drivers = []
    for d in (brief.get("drivers") or []):
        c = d.get("citation")
        ok = isinstance(c, int) and 1 <= c <= len(filings)
        drivers.append({
            "claim": d.get("claim", ""),
            "citation": c,
            "resolved": filings[c - 1]["headline"] if ok else None,
            "status": "ok" if ok else ("uncited" if c is None else "fabricated"),
        })

    return {
        "ticker": ticker, "cutoff": cutoff,
        "verdict": brief.get("verdict"), "confidence": brief.get("confidence"),
        "bull_case": brief.get("bull_case"), "bear_case": brief.get("bear_case"),
        "cited_figures": brief.get("cited_figures") or {},
        "l1_checks": (rec.get("l1") or {}).get("checks") or [],
        "drivers": drivers,
        "filings": filings,
        "filings_considered": considered,
        "filings_dropped_noise": noise,
        "raw_text": rec.get("raw_text", ""),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):        # keep the console readable
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/window"):
            from urllib.parse import parse_qs, urlparse

            q = parse_qs(urlparse(self.path).query)
            t = (q.get("ticker") or [""])[0]
            c = (q.get("cutoff") or [""])[0]
            try:
                body = json.dumps(window_detail(t, c)).encode("utf-8")
            except Exception as exc:                            # noqa: BLE001
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
                return self._send(500, body, "application/json")
            return self._send(200, body, "application/json")

        if self.path.startswith("/api/state"):
            try:
                body = json.dumps(cached_state()).encode("utf-8")
            except Exception as exc:                          # noqa: BLE001
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
                return self._send(500, body, "application/json")
            return self._send(200, body, "application/json")

        name = "index.html" if self.path in ("/", "") else self.path.lstrip("/")
        path = (STATIC / name).resolve()
        if not str(path).startswith(str(STATIC.resolve())) or not path.exists():
            return self._send(404, b"not found", "text/plain")
        ctype = {"html": "text/html; charset=utf-8", "css": "text/css",
                 "js": "text/javascript"}.get(path.suffix.lstrip("."), "text/plain")
        return self._send(200, path.read_bytes(), ctype)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.server")
    p.add_argument("--port", type=int, default=8600)
    args = p.parse_args(argv)

    print("loading frozen grid and baselines...", flush=True)
    _frozen.update(_load_frozen())
    print(f"  {_frozen['grid_stats']['ok']:,}/{_frozen['grid_stats']['expected']:,} grid cells")
    print(f"\n  dashboard -> http://localhost:{args.port}\n", flush=True)
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
