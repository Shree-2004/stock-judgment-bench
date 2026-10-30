"""M3: append today's predictions to the ledger, then witness the head in git.

    python -m scripts.live_predict              # run today's round (after 15:30 IST close)
    python -m scripts.live_predict --status     # what the ledger holds
    python -m scripts.live_predict --verify     # recompute the hash chain
    python -m scripts.live_predict --no-commit  # run, but leave ledger/chain.txt uncommitted

Runs only for *today's* session in IST, and only once ^NSEI has posted today's
close. That one rule ensures each prediction is made after the close it reads
and before the next open. A rerun the same evening fills in stocks that failed
or were cut off by the provider's daily quota. It never touches a stock that
already succeeded.

After the commit it settles calls whose 21-session window has closed (M4,
scripts.settle) and rebuilds site/data (scripts.export_site). Either step can
fail without affecting the round or the exit code.

Exit codes: 0 round complete · 1 nothing to do (holiday / prices not posted) ·
2 round partial or failed · 3 provider quota exhausted (rerun later).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from live import store
from live.filings import IST, NSE_NAMES
from live.ledger import (CHAIN_FILE, ROOT, Prediction, append, append_chain_line, done_symbols,
                         ensure_schema, evidence_for, finish_round, head_hash, start_round, verify_chain)

LOG = ROOT / "logs" / "live_predict.log"


def latest_session() -> date | None:
    """The date of the latest ^NSEI daily row: the NSE calendar, by construction."""
    import yfinance as yf
    hist = yf.Ticker("^NSEI").history(period="7d", auto_adjust=False)
    if hist is None or hist.empty:
        return None
    return hist.index[-1].date()


def run_round(conn, session: date, symbols, now, snapshot_fn, generate_fn, model: str,
              prompt_version: str) -> dict:
    """One pass over the stocks still missing a successful prediction for `session`."""
    from llm.client import DailyQuotaExhausted

    round_id = start_round(conn, session, now().isoformat(timespec="seconds"), model, prompt_version)
    todo = [s for s in symbols if s not in done_symbols(conn, session)]
    ok = failed = 0
    quota_hit = None

    for sym in todo:
        made_at = now().isoformat(timespec="seconds")
        bundle, kept = evidence_for(conn, sym, session, made_at)
        p = Prediction(symbol=sym, made_at=made_at, model=model, prompt_version=prompt_version, evidence=kept)
        try:
            snap = snapshot_fn(sym, session)
            p.price, p.snapshot = snap.price, snap.as_dict()
            result = generate_fn(snap, bundle)
            p.brief, p.raw_text = result.as_dict(), result.raw_text
            p.verdict, p.confidence = result.brief.verdict, result.brief.confidence
            if not result.brief.parsed:
                p.error = "parse: no verdict in response"
        except DailyQuotaExhausted as exc:
            quota_hit = str(exc)            # not this stock's fault: no row, retried on rerun
            break
        except Exception as exc:            # recorded, never skipped
            p.error = f"{type(exc).__name__}: {exc}"[:500]
        append(conn, round_id, session, p)
        ok, failed = (ok + 1, failed) if p.error is None else (ok, failed + 1)

    complete = not quota_hit and len(done_symbols(conn, session)) == len(symbols)
    state = "ok" if complete else ("failed" if ok == 0 and todo else "partial")
    if not todo:
        state = "ok"
    finish_round(conn, round_id, finished_at=now().isoformat(timespec="seconds"), state=state,
                 n_ok=ok, n_failed=failed, error=quota_hit)
    return {"round_id": round_id, "state": state, "attempted": ok + failed, "ok": ok,
            "failed": failed, "quota": quota_hit, "head": head_hash(conn)}


def commit_chain(session: date, state: str, head: str) -> str:
    """Commit only ledger/chain.txt, and only on master, so the witness never
    lands on a feature branch or picks up someone's half-staged work."""
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)

    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != "master":
        return f"not committed: on branch {branch!r}; the next run on master commits it"
    rel = CHAIN_FILE.relative_to(ROOT).as_posix()
    git("add", "--", rel)
    res = git("commit", "-m", f"ledger: {session} {state} head {head[:12]}", "--", rel)
    return "committed" if res.returncode == 0 else f"commit failed: {(res.stderr or res.stdout).strip()[:200]}"


def _soft(step) -> str:
    """Run a post-round step whose failure must not touch the round."""
    try:
        return step()
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}"[:200]


def refresh_site(write=None) -> str:
    """Rebuild site/data after a round. Runs after the ledger is committed, and
    a failure here is logged and nothing more: the site is a view, never part
    of the record."""
    if write is None:
        def write():
            from scripts.export_site import write_site
            return write_site()
    return _soft(write)


def settle_due(conn) -> str:
    """M4: settle calls whose 21-session window closed today. Appends to
    `outcomes` only; a failure is retried tomorrow."""
    def step():
        from scripts.settle import run
        return run(conn)
    return _soft(step)


def cmd_status(conn) -> int:
    rounds = conn.execute("SELECT session_date, state, n_ok, n_failed, head_hash FROM ledger_rounds "
                          "ORDER BY id DESC LIMIT 10").fetchall()
    total = conn.execute("SELECT COUNT(*), COUNT(DISTINCT session_date) FROM predictions").fetchone()
    print(f"{total[0]:,} ledger rows over {total[1]} session(s); head {head_hash(conn)[:16]}")
    for r in rounds:
        print(f"  {r[0]}  {r[1]:<7}  ok={r[2]:<3} failed={r[3]:<3} head={(r[4] or '')[:12]}")
    # README analysis plan item 6: a change of build behind the model name ends a track.
    builds = conn.execute(
        "SELECT session_date, model, GROUP_CONCAT(DISTINCT COALESCE("
        "json_extract(brief_json, '$.usage.model_version'), 'unrecorded')) "
        "FROM predictions WHERE verdict IS NOT NULL GROUP BY session_date, model "
        "ORDER BY session_date DESC LIMIT 10").fetchall()
    print("model builds by session:")
    for session, model, seen in builds:
        flag = "   <-- more than one build in a session" if "," in (seen or "") else ""
        print(f"  {session}  {model}  {seen}{flag}")
    return 0


def cmd_verify(conn) -> int:
    ok, n, bad = verify_chain(conn)
    print(f"chain {'OK' if ok else 'BROKEN at seq ' + bad}: {n:,} rows checked")
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-commit", action="store_true")
    args = ap.parse_args(argv)

    conn = store.connect(args.db)
    ensure_schema(conn)
    if args.status:
        return cmd_status(conn)
    if args.verify:
        return cmd_verify(conn)

    from config import settings
    from data.market_data import get_market_snapshot
    from llm.client import flush_traces, generate_brief
    from llm.prompts import DEFAULT_VERSION

    today = datetime.now(IST).date()
    session = latest_session()
    if session != today:
        print(f"no round: today is {today}, latest NSE session is {session} "
              "(holiday, weekend, or today's close not posted yet)")
        return 1

    model = settings.llm_model
    summary = run_round(
        conn, session, list(NSE_NAMES), now=lambda: datetime.now(timezone.utc),
        snapshot_fn=lambda sym, s: get_market_snapshot(sym, as_of=s),
        generate_fn=lambda snap, bundle: generate_brief(snap, bundle, prompt_version=DEFAULT_VERSION, model=model),
        model=model, prompt_version=DEFAULT_VERSION)
    flush_traces()

    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    append_chain_line(CHAIN_FILE, session, summary["state"], total, summary["head"], stamp)
    witness = "skipped (--no-commit)" if args.no_commit else commit_chain(session, summary["state"], summary["head"])
    settled = settle_due(conn)
    site = refresh_site()

    line = (f"{stamp} session={session} state={summary['state']} attempted={summary['attempted']} "
            f"ok={summary['ok']} failed={summary['failed']} head={summary['head'][:12]} git={witness}"
            + (f" quota={summary['quota'][:80]!r}" if summary["quota"] else "")
            + f" settle={settled!r} site={site!r}")
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    if summary["quota"]:
        return 3
    return 0 if summary["state"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
