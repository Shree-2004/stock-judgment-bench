"""M2: check the rule classifier against hand labels.

    python -m scripts.label_filings stats     # category mix over the whole store
    python -m scripts.label_filings sample    # draw the 50-filing label sheet (once)
    python -m scripts.label_filings score     # compare filled-in labels with the rules

**The sample is held out.** Only filings first seen after `RULES_FROZEN_AT`
are eligible, so the rules are scored on text they were not written against.
The sheet does not show the rule's answer, so it cannot anchor the labeller.

**Two strata, reported separately.**
- `random` (30): a simple random draw from company filings. Its accuracy is an
  honest estimate for the feed as a whole.
- `hard` (20): drawn only from filings with a generic NSE tag ("Updates",
  "Press Release", ...) where the rules have to read free text. This is where
  the rules can actually fail; mixing it into the headline would understate
  overall accuracy, leaving it out would hide the weak spot.

Fund NAV declarations are excluded from both by their NSE tag — they are not
company filings and would spend a fifth of the labels on a trivial case. The
filter is the raw tag, not the classifier's output, so it cannot hide a
classifier mistake.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from live import store
from live.classify import (CATEGORIES, CLASSIFIER_VERSION, GENERIC_TAGS, RULES_FROZEN_AT,
                           classify, rules_fingerprint, split_subject)
from live.filings import IST

ROOT = Path(__file__).resolve().parent.parent
LABEL_DIR = ROOT / "labels"
SHEET = LABEL_DIR / "m2_label_sheet.csv"
SAMPLE = LABEL_DIR / "m2_sample.json"
SCORE = LABEL_DIR / "m2_score.json"

SEED = 20261029
N_RANDOM, N_HARD = 30, 20
COLUMNS = ["n", "provider_ref", "published_ist", "company", "description", "url", "label", "notes"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; sane at n=20 and at k=n, unlike the normal one."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_rows(db: Path) -> list[dict]:
    conn = store.connect(db, read_only=True)
    return [dict(r) for r in conn.execute(
        "SELECT provider_ref, company, subject, url, published_at, first_seen_at "
        "FROM filings ORDER BY id")]


# ---------------------------------------------------------------------------

def cmd_stats(args) -> int:
    rows = load_rows(args.db)
    labels = [classify(r["subject"]) for r in rows]
    cats = Counter(c.category for c in labels)
    print(f"{len(rows):,} filings, rules {CLASSIFIER_VERSION}\n")
    for cat in CATEGORIES:
        print(f"  {cats.get(cat, 0):6,}  {cat}")
    via_text = sum(not c.rule.startswith("tag.") for c in labels)
    print(f"\n  decided by tag alone: {len(rows) - via_text:,}   "
          f"needed free text: {via_text:,}   unmatched tag: "
          f"{sum(c.rule == 'unmatched_tag' for c in labels):,}")
    held_out = sum(r["first_seen_at"] > RULES_FROZEN_AT for r in rows)
    print(f"  first seen after the rules froze ({RULES_FROZEN_AT}): {held_out:,}")
    return 0


def cmd_sample(args) -> int:
    if SHEET.exists() and not args.force:
        print(f"{SHEET.relative_to(ROOT)} already exists; the sample is drawn once. "
              "--force redraws it and throws away any labels in it.")
        return 1

    pool = [r for r in load_rows(args.db)
            if r["first_seen_at"] > RULES_FROZEN_AT
            and split_subject(r["subject"])[1] != "declaration of nav"]
    hard_pool = [r for r in pool if split_subject(r["subject"])[1] in GENERIC_TAGS]
    print(f"eligible: {len(pool):,} company filings first seen after {RULES_FROZEN_AT}, "
          f"{len(hard_pool):,} with a generic tag")
    if len(pool) < N_RANDOM + N_HARD or len(hard_pool) < N_HARD:
        print(f"not enough yet (need {N_RANDOM + N_HARD} and {N_HARD}); "
              "let the poller run and try again later.")
        return 1

    rng = random.Random(SEED)
    hard = rng.sample(hard_pool, N_HARD)
    taken = {r["provider_ref"] for r in hard}
    rand = rng.sample([r for r in pool if r["provider_ref"] not in taken], N_RANDOM)

    # Interleave so the labeller cannot tell which stratum a row came from.
    rows = [("hard", r) for r in hard] + [("random", r) for r in rand]
    rng.shuffle(rows)

    LABEL_DIR.mkdir(exist_ok=True)
    with SHEET.open("w", newline="", encoding="utf-8-sig") as f:    # BOM: Excel reads ₹ correctly
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for n, (_, r) in enumerate(rows, 1):
            w.writerow({
                "n": n,
                "provider_ref": r["provider_ref"],
                "published_ist": datetime.fromisoformat(r["published_at"]).astimezone(IST)
                                         .strftime("%Y-%m-%d %H:%M"),
                "company": r["company"],
                "description": r["subject"],
                "url": r["url"] or "",
                "label": "",
                "notes": "",
            })
    SAMPLE.write_text(json.dumps({
        "classifier_version": CLASSIFIER_VERSION,
        "rules_fingerprint": rules_fingerprint(),
        "rules_frozen_at": RULES_FROZEN_AT,
        "seed": SEED,
        "pool_size": len(pool),
        "hard_pool_size": len(hard_pool),
        "strata": {s: [r["provider_ref"] for t, r in rows if t == s] for s in ("random", "hard")},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SHEET.relative_to(ROOT)} ({len(rows)} rows) and {SAMPLE.relative_to(ROOT)}")
    print("fill the `label` column using labels/README.md, then run `score`.")
    return 0


def cmd_score(args) -> int:
    sheet_path = args.sheet or SHEET
    if not sheet_path.exists():
        print(f"no label sheet at {sheet_path}; run `sample` first.")
        return 1
    meta = json.loads(SAMPLE.read_text(encoding="utf-8"))
    stratum = {ref: s for s, refs in meta["strata"].items() for ref in refs}
    with sheet_path.open(encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))

    foreign = [r["n"] for r in sheet if r["provider_ref"] not in stratum]
    if foreign or len(sheet) != len(stratum):
        print(f"{sheet_path.name} is not the drawn sample (rows not in it: {foreign or 'count differs'})")
        return 1
    score_path = SCORE if args.labeller == "human" else SCORE.with_name(f"m2_score_{args.labeller}.json")
    if args.labeller != "human":
        print(f"labeller: {args.labeller} — this measures agreement with that labeller, "
              "not accuracy against hand labels.")

    bad = [(r["n"], r["label"]) for r in sheet
           if r["label"].strip() and r["label"].strip() not in CATEGORIES]
    if bad:
        print("labels not in the taxonomy (see labels/README.md):")
        for n, lab in bad:
            print(f"  row {n}: {lab!r}")
        return 1

    done = [r for r in sheet if r["label"].strip()]
    held_out = meta["rules_fingerprint"] == rules_fingerprint()
    print(f"{len(done)} of {len(sheet)} rows labelled; scoring rules {CLASSIFIER_VERSION} "
          f"({rules_fingerprint()})")
    if not held_out:
        print(f"  NOTE: the sample was drawn for rules {meta['rules_fingerprint']}. The rules "
              "changed after it was drawn, so this score is in-sample, not held-out.")
    if not done:
        return 1

    results, wrong = {}, []
    for r in done:
        pred = classify(r["description"])
        gold = r["label"].strip()
        r["_pred"], r["_rule"], r["_ok"] = pred.category, pred.rule, pred.category == gold
        if not r["_ok"]:
            wrong.append(r)

    for name in ("random", "hard", "all"):
        rows = [r for r in done if name == "all" or stratum.get(r["provider_ref"]) == name]
        k = sum(r["_ok"] for r in rows)
        lo, hi = wilson(k, len(rows))
        results[name] = {"n": len(rows), "correct": k,
                         "accuracy": round(k / len(rows), 4) if rows else None,
                         "ci95": [round(lo, 4), round(hi, 4)]}
        if rows:
            print(f"  {name:6}  {k:2}/{len(rows):<2}  {k / len(rows):6.1%}   "
                  f"95% CI {lo:.1%} – {hi:.1%}")

    confusion = Counter((r["label"].strip(), r["_pred"]) for r in wrong)
    if wrong:
        print("\ndisagreements (gold <- predicted, rule):")
        for r in wrong:
            print(f"  #{r['n']:>2} {r['label'].strip():19} <- {r['_pred']:19} {r['_rule']:24} "
                  f"{r['company'][:30]}")

    score_path.write_text(json.dumps({
        "labeller": args.labeller,
        "sheet": sheet_path.name,
        "classifier_version": CLASSIFIER_VERSION,
        "rules_fingerprint": rules_fingerprint(),
        "held_out": held_out,
        "labelled": len(done),
        "of": len(sheet),
        "strata": results,
        "confusion": [{"gold": g, "predicted": p, "count": c} for (g, p), c in confusion.most_common()],
        "disagreements": [{"n": int(r["n"]), "provider_ref": r["provider_ref"], "gold": r["label"].strip(),
                           "predicted": r["_pred"], "rule": r["_rule"]} for r in wrong],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {score_path.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    sp = sub.add_parser("sample")
    sp.add_argument("--force", action="store_true")
    sc = sub.add_parser("score")
    sc.add_argument("--sheet", type=Path, help="labelled sheet (default: the hand-label sheet)")
    sc.add_argument("--labeller", choices=("human", "ai_tool"), default="human",
                    help="who filled it in; recorded in the score so it cannot be misreported")
    args = ap.parse_args(argv)
    return {"stats": cmd_stats, "sample": cmd_sample, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
