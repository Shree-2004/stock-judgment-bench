"""Generate every brief in the benchmark. Long, resumable, restartable.

Groq's binding limit is 8,000 tokens/minute account-wide (measured across all
four available models — they share one budget). A brief costs ~1,400 tokens, so
the ceiling is ~5.7 briefs/min and a full 3,690-brief run takes ~11 hours. That
is a background job, not something to sit and watch.

Resumability is the same discipline as the disclosure cache: one JSON file per
(prompt_version, symbol, cutoff), written atomically, skipped if present. Kill
it and rerun whenever; the only cost of an interruption is the call in flight.

Failures are recorded as files with `error` set, never skipped silently. A
window missing from the results because the API refused is not the same as a
window where the model produced a bad brief, and merging them would corrupt the
grounding rate this benchmark exists to measure.

    python -m scripts.run_briefs --limit 20          # smoke test
    python -m scripts.run_briefs                     # the full run
    python -m scripts.run_briefs --prompt-version v2_prose
    python -m scripts.run_briefs --retry-failed
    python -m scripts.run_briefs --status
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

from config import settings
from data.market_data import MarketDataError
from data.snapshot_cache import get_snapshot_cached
from data.news_data import DisclosureCacheMiss, get_disclosures
from data.universe import UNIVERSE
from data.windows import build_windows
from llm.client import (DailyQuotaExhausted, LLMUnavailableError, flush_traces,
                        generate_brief, get_langfuse)
from llm.prompts import DEFAULT_VERSION

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "cache" / "briefs"


def model_slug(model: str) -> str:
    """Results are namespaced by model AND decoding config.

    A benchmark measures one configuration. Mixing gpt-oss-120b with
    gemini-2.5-flash would obviously blend two subjects — but so does mixing
    the same model at different reasoning efforts, since effort changes what
    the model actually does (790 vs 359 completion tokens on an identical
    prompt). The effort suffix keeps those runs in separate namespaces instead
    of silently averaging them.
    """
    slug = model.replace("/", "_").replace(":", "_")
    effort = getattr(settings, "reasoning_effort", "") or ""
    if effort and "gpt-oss" in model:
        slug = f"{slug}@{effort}"
    return slug


def result_path(prompt_version: str, model: str, symbol: str,
                cutoff: date | str) -> Path:
    cutoff_s = cutoff if isinstance(cutoff, str) else cutoff.isoformat()
    safe = symbol.replace("&", "_AND_").replace("/", "_")
    return RESULTS_ROOT / model_slug(model) / prompt_version / safe / f"{cutoff_s}.json"


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def status(prompt_version: str, model: str) -> dict:
    root = RESULTS_ROOT / model_slug(model) / prompt_version
    counts = {"files": 0, "ok": 0, "failed": 0, "unparsed": 0, "unreadable": 0}
    grounding: list[float] = []
    clean = 0
    verdicts: dict[str, int] = {}
    tokens = 0

    if not root.exists():
        return counts

    for p in root.rglob("*.json"):
        counts["files"] += 1
        d = _load(p)
        if d is None:
            counts["unreadable"] += 1
            continue
        if d.get("error"):
            counts["failed"] += 1
            continue
        counts["ok"] += 1
        brief = d.get("brief") or {}
        if not brief.get("parsed"):
            counts["unparsed"] += 1
        v = brief.get("verdict")
        if v:
            verdicts[v] = verdicts.get(v, 0) + 1
        l1 = d.get("l1") or {}
        if l1.get("score") is not None:
            grounding.append(float(l1["score"]))
        if l1.get("clean"):
            clean += 1
        tokens += (d.get("usage") or {}).get("total_tokens", 0) or 0

    counts["verdicts"] = verdicts
    counts["mean_l1_grounding"] = round(sum(grounding) / len(grounding), 4) if grounding else None
    counts["clean_rate"] = round(clean / counts["ok"], 4) if counts["ok"] else None
    counts["total_tokens"] = tokens
    return counts


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="python -m scripts.run_briefs")
    p.add_argument("--prompt-version", default=DEFAULT_VERSION)
    p.add_argument("--model", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--no-trace", action="store_true")
    p.add_argument("--status", action="store_true")
    args = p.parse_args(argv)

    if args.status:
        print(json.dumps(status(args.prompt_version, args.model or settings.llm_model), indent=2))
        return 0

    model = args.model or settings.llm_model
    windows = build_windows()
    # Date-major, NOT company-major. L3 scores one observation per date from
    # the cross-section of names on that date, so a run interrupted at hour 6
    # must leave complete cross-sections for the dates it reached rather than
    # a few tickers smeared thinly across all 90 dates (which scores nothing).
    cells = [(c, w.cutoff) for w in windows for c in UNIVERSE]

    todo = []
    for company, cutoff in cells:
        existing = _load(result_path(args.prompt_version, model, company.symbol, cutoff))
        if existing and not (existing.get("error") and args.retry_failed):
            continue
        todo.append((company, cutoff))

    print(f"prompt={args.prompt_version}  model={model}")
    print(f"tracing: {'on' if not args.no_trace and get_langfuse() else 'off'}")
    print(f"{len(cells) - len(todo):,} already done, {len(todo):,} to generate", flush=True)
    if not todo:
        print(json.dumps(status(args.prompt_version, model), indent=2))
        return 0

    if args.limit:
        todo = todo[: args.limit]
        print(f"--limit: generating {len(todo):,}", flush=True)

    started = time.monotonic()
    counts = {"ok": 0, "failed": 0, "unparsed": 0}

    for i, (company, cutoff) in enumerate(todo, 1):
        path = result_path(args.prompt_version, model, company.symbol, cutoff)
        record: dict

        try:
            snapshot = get_snapshot_cached(company.ticker, cutoff, include_name=True)
            disclosures = get_disclosures(company.symbol, cutoff, limit=settings.disclosure_limit)
            result = generate_brief(
                snapshot, disclosures,
                prompt_version=args.prompt_version,
                model=model,
                trace=not args.no_trace,
            )
            record = result.as_dict()
            record["raw_text"] = result.raw_text
            counts["ok"] += 1
            if not result.brief.parsed:
                counts["unparsed"] += 1

        except DailyQuotaExhausted as exc:
            # Stop, do not mark the rest failed. Every remaining cell is fine;
            # the provider is simply done for the day. Rerun tomorrow.
            print(f"\n  daily quota reached after {i - 1} briefs this run: {exc}")
            print(f"  {counts['ok']} generated. Rerun tomorrow to continue.", flush=True)
            flush_traces()
            print(json.dumps(status(args.prompt_version, model), indent=2))
            # Exit code 3 means "the provider is done for the day", distinct
            # from 0 ("everything generated") and from a crash. The daily
            # wrapper loops until it sees 3 or 0, so a process killed mid-run
            # simply restarts and resumes — the cache makes that free.
            return 3

        except (MarketDataError, DisclosureCacheMiss, LLMUnavailableError) as exc:
            # Recorded, not skipped: "the pipeline refused" must stay
            # distinguishable from "the model wrote something wrong".
            record = {
                "ticker": company.ticker,
                "cutoff": cutoff.isoformat(),
                "prompt_version": args.prompt_version,
                "model": model,
                "error": f"{type(exc).__name__}: {exc}",
            }
            counts["failed"] += 1

        record["generated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_atomic(path, record)

        if i % 10 == 0 or i == len(todo):
            elapsed = time.monotonic() - started
            eta = (elapsed / i) * (len(todo) - i)
            print(
                f"  [{i:>5}/{len(todo)}] {company.symbol:<11} {cutoff}  "
                f"ok={counts['ok']} failed={counts['failed']} unparsed={counts['unparsed']}  "
                f"elapsed={_fmt(elapsed)} eta={_fmt(eta)}",
                flush=True,
            )
            flush_traces()

    flush_traces()
    print(f"\ndone at {datetime.now():%H:%M:%S}")
    print(json.dumps(status(args.prompt_version, model), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
