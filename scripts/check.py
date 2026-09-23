"""Is the benchmark finished? One command, one clear answer.

Every stage has its own cache and its own notion of "complete", so without
this you have to remember four different status commands and compare counts by
eye. This prints a checklist and a verdict.

    python -m scripts.check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from config import settings
from data.universe import UNIVERSE
from data.windows import build_windows
from evals.grid import grid_stats
from scripts.run_briefs import model_slug, status as brief_status

ROOT = Path(__file__).resolve().parent.parent
OK, NO, PART = "[done]", "[missing]", "[partial]"


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    windows = build_windows()
    tickers = [c.ticker for c in UNIVERSE]
    total = len(tickers) * len(windows)
    rows, done_all = [], True

    # 1. grid
    g = grid_stats(tickers, [w.cutoff for w in windows])
    ok = g["complete"]
    done_all &= ok
    rows.append((OK if ok else PART, "price grid",
                 f"{g['ok']:,}/{g['expected']:,} cells"))

    # 2. disclosures
    dis = ROOT / "cache" / "bse"
    n_dis = sum(1 for _ in dis.rglob("*.json")) if dis.exists() else 0
    ok = n_dis >= total
    done_all &= ok
    rows.append((OK if ok else PART, "BSE disclosures", f"{n_dis:,}/{total:,} windows"))

    # 3. baselines
    bl = ROOT / "cache" / "baselines.json"
    ok = bl.exists()
    detail = "not computed"
    if ok:
        try:
            d = json.loads(bl.read_text(encoding="utf-8"))
            ok = not d.get("partial", False)
            detail = f"{len(d.get('results', {}))} strategies" + ("" if ok else " (PARTIAL)")
        except (ValueError, OSError):
            ok, detail = False, "unreadable"
    done_all &= ok
    rows.append((OK if ok else NO, "baselines", detail))

    # 4. briefs
    model = settings.llm_model
    bs = brief_status("v1_structured", model)
    n_ok, n_fail = bs.get("ok", 0), bs.get("failed", 0)
    ok = n_ok >= total
    done_all &= ok
    pct = n_ok / total * 100 if total else 0
    rows.append((OK if ok else PART, f"briefs ({model})",
                 f"{n_ok:,}/{total:,} ({pct:.1f}%)"
                 + (f", {n_fail} failed" if n_fail else "")))

    width = max(len(r[1]) for r in rows) + 2
    print(f"\nstock-judgment-bench — {model}\n")
    for mark, name, detail in rows:
        print(f"  {mark:<10} {name:<{width}} {detail}")

    print()
    if done_all:
        print("  ALL STAGES COMPLETE — run the report:\n")
        print("      python -m scripts.report\n")
        return 0

    remaining = max(0, total - n_ok)
    days = remaining / 500
    print(f"  NOT FINISHED — {remaining:,} briefs left")
    if remaining:
        print(f"  At 500/day (the free-tier cap) that is ~{days:.1f} more days.")
        print("  The scheduled task 'sjb-briefs' runs daily at 12:35.")
    print()
    print("  Interim numbers (clearly marked partial):")
    print("      python -m scripts.report --allow-incomplete\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
