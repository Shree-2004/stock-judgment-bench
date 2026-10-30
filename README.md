# stock-judgment-bench

**Does an LLM's stock judgement contain anything, once lookahead leakage and market drift are removed?**

Indian equities, NSE, 2019–2026. The expected answer is *no* — and that is the point. A leak-free harness reporting "no detectable edge" is worth more than a repo claiming it predicts stocks, because every one of those is either leaking future data or not checking.

The brief generator is the thing under test, not the product.

## How it works

Pick a date in the past. Build a snapshot containing only what was knowable that day, plus the company's exchange filings from the preceding three weeks. Hand it to the model, ask for a verdict. Then score three things:

| | Check | Ground truth | Status |
|---|---|---|---|
| **L1** | Are the cited numbers right? | `MarketSnapshot.figures()` | works |
| **L2** | Are the cited *reasons* real? | the BSE filings supplied | tier 1 works; tier 2 falsified |
| **L3** | Was the call right? | forward 21-trading-day return | works — no edge found |

**L3 is scored cross-sectionally**: within each date, do the model's bullish names outperform its bearish ones? That is market-neutral by construction — which is why `always_bullish`, the strategy that would look brilliant on raw returns, is *unscoreable* here rather than strong.

One date is one observation. 3,690 briefs do not become 3,690 degrees of freedom; they become 90.

## Results

**Complete.** 3,690 briefs, 41 NSE large caps, 90 non-overlapping 21-trading-day windows, 2019–2026.

### L3 — does the model's judgement predict anything?

| strategy | dates | rank corr | t | spread % | hit |
|---|---|---|---|---|---|
| **model** (`gemini-3.1-flash-lite`) | 90 | **−0.0153** | **−0.65** | −0.885 | 44.4% |
| always_bullish | 90 | — | — | — | — |
| random | 90 | +0.0123 | 0.72 | +0.211 | 53.3% |
| momentum | 90 | −0.0467 | −1.88 | −0.916 | 38.2% |
| reversal | 90 | +0.0467 | +1.88 | +0.916 | 61.8% |

**No detectable edge.** t = −0.65 across 90 independent dates — indistinguishable from random.

The estimate wandered −0.008 → +0.010 → −0.013 → −0.015 as data accumulated, twice changing sign. That drift around zero is stronger evidence than any single reading.

`always_bullish` reports nothing on every metric. Under cross-sectional scoring its predictor is constant, so it is *unscoreable* rather than merely bad — the design making market drift unexploitable, which is the only reason the other numbers mean anything.

### The pre-registered prediction was neither confirmed nor ruled out

Recorded before any brief existed: an early 16-ticker probe showed verdicts tracking one-month momentum, so if the model were merely re-reading momentum it should land near **−0.047**.

Final: **−0.0153**, about a third of that and not distinguishable from zero. It is not distinguishable from −0.047 either. The standard error is 0.0235, so the 95% interval is **−0.061 to +0.031**. That interval contains both zero and the predicted −0.047, and the gap from the prediction is t = 1.34.

So the benchmark answers the main question (no detectable edge) but cannot decide the mechanism. With 90 dates, the smallest correlation it can reliably detect is about |0.066| at 80% power, and the momentum and reversal baselines (±0.047) sit below that. An earlier version of this README said the prediction was "falsified". The numbers above do not support that, so the claim was withdrawn.

### L1 — numeric grounding

| | |
|---|---|
| briefs | 3,690 (0 failed, 0 unparsed) |
| mean grounding | **0.9998** |
| clean rate | 0.9989 |
| briefs with a wrong figure | **4** |
| briefs with an invented label | **0** |

Four bad numbers in **17,628** cited figures — and **two are the same bug**:

```
NESTLEIND 2023-06-07   week52_low  cited 1,664,240   truth 16,642.40
NESTLEIND 2023-07-07   week52_low  cited 1,779,505   truth 17,795.05
TITAN     2020-09-18   change_pct_1d  cited  +3.37   truth  −1.59
BAJFINANCE 2026-06-03  change_pct_1d  cited  −8.53   truth  −0.59
```

The two NESTLEIND failures are a ×100 decimal shift on the same field of the same stock — the one name in the universe trading in five figures. The TITAN case is a sign inversion: the model reported a 3.4% rise on a day the stock fell 1.6%.

A substring-based grounding check would likely have passed the TITAN case, since `3.37` plausibly appears somewhere in a nine-figure snapshot. Field-aware checking is what catches a right-looking number attached to the wrong claim.

### L2 — attribution (tier 1)

| | |
|---|---|
| claims | **8,804** |
| fabricated citations | **0** |
| uncited claims | 1 |
| citation validity | **0.9999** |

Zero phantom filings across 8,804 claims.

### What the three layers together show

**The model is near-perfect at handling the data and worthless at judging it.** It copies figures accurately, never invents a source, cites correctly — and its verdicts predict nothing.

That separation is only visible because L1, L2 and L3 were measured independently. A single "quality score" would have averaged a 0.9998 and a −0.0153 into something meaningless.

## Status

| Step | What | State |
|---|---|---|
| 1 | Point-in-time data layer + leak tests | done |
| 2 | BSE disclosure cache + selection rule | done |
| 3 | Brief generator, L1 grounding, Langfuse tracing | done |
| 4 | Grid, baselines, L3 cross-sectional scoring | done |
| 5 | L2 attribution — tier 1 done, tier 2 falsified | partial |
| 6 | Memorisation control, prose-vs-structured ablation | not started |
| — | **Full 3,690-brief run** | **done** — frozen as tag `v1.0` |
| 2.0 / M1 | Live filing watcher (NSE RSS → SQLite → dashboard) | done |
| 2.0 / M2 | Rule-based filing categories, scored on a held-out sample | v1: 72% agreement with AI-tool labels; v2 (held-out): 88% (44/50) |
| 2.0 / M3 | Forward ledger: 41 stocks daily, hash-chained, witnessed in git | built; first round 2026-10-29; second track (gpt-oss-120b) from 2026-10-30 |
| 2.0 / M4 | Settlement + the pre-registered analysis | built and tested (v1 replay reproduces −0.0153 / t −0.65); first calls settle ~2026-11-27 |

## Live: forward-only record (2.0)

The v1 result is historical, and a model may have seen 2019–2026 in training. The only clean memorisation control is a forward record: predictions made before the outcome exists, stored so they cannot be edited. M1 is the first piece — the evidence feed those predictions will read.

**M1 — filing watcher.** Polls NSE's official announcements RSS every 15 minutes and stores every item, exchange-wide, in `store/live.sqlite`.

| Rule | How |
|---|---|
| A filing is stored once | `UNIQUE(provider, provider_ref)`; ref = hash of company, time, link, subject |
| Stored filings never change | SQLite triggers abort any `UPDATE` or `DELETE` on `filings` |
| Two clocks, never merged | `published_at` (exchange) and `first_seen_at` (this system). A prediction may only use a filing it had *seen* by its cutoff |
| Every poll is logged | `job_runs`: fetched / inserted / duplicate / rejected / attempts / error |
| Lost coverage is admitted | if a poll's oldest item is newer than everything stored, the gap is recorded |

**Why NSE, not BSE.** On 2026-10-28 BSE's API returned an Akamai *Access Denied* to every request, historical windows included — an edge block. It is not worked around. NSE's RSS answers a plainly-identified client (a UA containing a URL is stalled until timeout; a plain one answers in 0.1 s).

**What the feed turned out to be** — measured, and each one is a fixture in `tests/live/`:

- a rolling window, not an archive: ~1,050 items covering **~6 hours** on a results-season evening. Nothing can be backfilled, hence the 15-minute poll and gap detection
- no stable ID: one PDF link appeared 8 times at different seconds; 27 items had no link
- two date shapes (`17:44:30`, `18:56`), occasionally missing; IST with no offset
- items vanish from the middle: between two polls 90 s apart the feed shrank by 20 while its oldest item stayed put

First live runs, 2026-10-28:

```
job 1  fetched 1070  inserted 1064  duplicate    5  rejected 1
job 2  fetched 1070  inserted    0  duplicate 1069  rejected 1    <- identical rerun: zero new rows
```

```powershell
python -m scripts.live_ingest                 # one poll
python -m scripts.live_ingest --unmatched     # check company-name mapping
streamlit run app/live_dashboard.py           # read-only view
powershell -ExecutionPolicy Bypass -File scripts\register_live_task.ps1   # poll every 15 min
```

**M2 — what kind of filing is it?** Each filing gets one of 15 categories (`live/classify.py`), decided by one question: *what new information does this give a holder?* A newspaper reprint of results is routine compliance, not results.

- The NSE subject tag settles **1,216 of the first 1,474** filings (82%). The rest carry a tag that says nothing ("Updates", "Press Release", "Outcome of Board Meeting"), and those are read from the free text.
- A filing that says nothing ("X has informed the Exchange about General Updates") is labelled `other`, not guessed at.
- Every label names the rule that produced it, and labels are computed on read. The store stays append-only, and fixing a rule never rewrites history.

**The rules are scored on text they never saw.** They were written against filings seen up to `RULES_FROZEN_AT` (2026-10-29 12:00 IST). The 50-filing hand-label sample is drawn only from filings first seen after that time. The sheet doesn't show the rules' answer, the draw is seeded and done once, and the sample records a fingerprint of the rules. If any rule changes after labelling, `score` reports the result as in-sample.

| stratum | n | what it measures |
|---|---|---|
| `random` | 30 | accuracy on company filings as a whole (NAVs excluded by raw tag) |
| `hard` | 20 | accuracy where it can actually fail: generic tags only |

Accuracy is reported per stratum with a Wilson 95% interval. At n = 30, even 30/30 only bounds accuracy above about 89%.

**Result (rules-v1, held-out sample of 2026-10-29).** The 50 rows were labelled by an **AI tool, not by hand**, so this is *agreement with a model labeller*, not accuracy against human ground truth:

| stratum | agree | 95% CI |
|---|---|---|
| random | 22/30 (73%) | 56–86% |
| hard | 14/20 (70%) | 48–86% |
| all | 36/50 (72%) | 58–83% |

The 14 disagreements, read one by one:

- **6: the description is empty, and the answer is only in the PDF's filename.** Examples: `Updates |SUBJECT: Updates` linking to `…Investor_Presentation.pdf`, and "General Updates" linking to `…Buyback_Undertaking.pdf`. The rules read only the description, so they said `other`; the tool read the link.
- **2: the code contradicts its own definition.** A board-meeting "outcome" that names no decision is defined as `board_meeting`, but the empty-text check runs first and sends it to `other`.
- **2: rule misses.** "Axiro Semiconductor *Invests* in ThinSiC" isn't caught as a deal, and a truncated "disclosure under Regulation 30…" falls through to `business_update` when it should be `other`.
- **4: the tool departs from the labelling guide.** It filed a newspaper reprint, a valuation report and two Reg 6(1)/7(1) filings as `other`; the guide puts them in routine compliance.

So by the guide's own definitions, about 10 of the 50 are rule errors, and most of those come from one gap: the PDF filename carries information the rules don't read. Fixes belong in rules-v2, which must be scored on a **new** sample drawn after its own freeze. Re-scoring this sample after fixing it would be in-sample.

**rules-v2**, frozen at 2026-10-29 16:00 IST, makes these changes:
- reads the PDF filename, but only when the description says nothing
- labels an outcome that names no decision `board_meeting`
- adds "invests" to the deal rule
- treats the statutory Regulation 30 preamble as boilerplate
- maps 10 NSE tags that first appeared after the v1 freeze

It is not re-scored on the v1 sample. Its own sample comes from filings first seen after 16:00 IST, and its files live in `labels/rules-v2/`.

**Result (rules-v2, held-out sample of 2026-10-29 evening).** The sample was drawn and committed (`c863284`) before labelling, and the labels were committed (`fe2326d`) before scoring. The labeller was again an **AI tool**, so this too is agreement, not accuracy:

| stratum | agree | 95% CI |
|---|---|---|
| random | 28/30 (93%) | 79–98% |
| hard | 16/20 (80%) | 58–92% |
| all | 44/50 (88%) | 76–94% |

Of the 6 disagreements, 4 are rows where the tool could not open the PDF and declined to infer anything from its filename (it said `other`), while the rules used the filename. The other 2 are judgement calls: an alteration of the company's main objects (tool: `business_update`, rules: routine compliance) and a debenture trustee's NOC (tool: `other`, rules: routine compliance).

This number is final for rules-v2; the rules will not be tuned on it. It is a better result than v1's 72%, but the two samples were drawn from different hours' filings and their intervals overlap, so the size of the improvement is not pinned down.

```powershell
python -m scripts.label_filings stats     # category mix
python -m scripts.label_filings sample    # draw the sheet once enough post-freeze filings exist
python -m scripts.label_filings score     # after filling labels/<rules-version>/label_sheet.csv (guide: labels/README.md)
```

**M3 — the forward ledger.** Every trading day after the close, one prediction per stock is made for all 41 stocks, with the same prompt (`v1_structured`) and model as v1. Each one is appended to `predictions` in `store/live.sqlite` (`live/ledger.py`). Its outcome doesn't exist yet, so no model can have seen it in training.

| Rule | How |
|---|---|
| Rows never change | SQLite triggers abort `UPDATE` and `DELETE` |
| Tampering is detectable | each row stores `prev_hash` and `entry_hash = sha256(prev_hash + row)`, and `--verify` recomputes the chain. A test edits a row behind the triggers' back and checks that the break is caught |
| Git witnesses it | after each round, the head hash is appended to `ledger/chain.txt` and committed on master. Once pushed, rewriting a prediction means rewriting public history |
| No stale prices | a round runs only if ^NSEI has posted *today's* close (IST). It is never silently based on yesterday's |
| Point-in-time evidence | only filings with `first_seen_at` ≤ the prediction's `made_at`, from the last 21 days. Routine compliance and NAVs are dropped, and the newest 5 distinct headlines are kept (v1's rule) |
| Failures are rows | an error is recorded with its evidence. A rerun appends `attempt + 1` and never touches a stock that succeeded. A quota stop leaves no row for the interrupted stock |

**Analysis plan, fixed 2026-10-29, before any outcome exists:**

1. *Outcome:* the 21-trading-session forward return from the close the prediction read, with the same `evals/outcomes.py` and `evals/scoring.py` as v1 (M4 must reproduce v1's numbers when replayed on v1 data).
2. *Excluded:* warm-up sessions before **2026-11-18**. The filing store began 2026-10-28, so earlier rounds can't have 21 days of evidence. Also excluded is any row made after the next session opened.
3. *Primary:* the mean per-session Spearman correlation between verdict and forward return. It is computed over **non-overlapping** sessions: the first non-warm-up session, then every 21st session after it. This is v1's metric, so the two are directly comparable.
4. *Secondary:* the same over every session, with a standard error that accounts for overlap. It is reported alongside the primary, never instead of it.
5. *Power, stated now:* v1's 90 dates gave a detection floor of about |0.066|. The primary series adds about 12 dates a year, so the ledger can't confirm or rule out an edge of v1's size for years. Its first job is the memorisation control: whether the post-training-cutoff correlation differs from v1's in-sample −0.015.
6. *Model continuity (added 2026-10-30, still before any outcome exists).* A track is one model name plus one prompt version, and it is never silently continued on a different model. A track ends at its last successful session when either of these happens:
   - the provider retires the model, meaning two consecutive sessions in which every call fails with not-found, deprecated, or permission errors
   - the build behind the name changes: Gemini returns `modelVersion` with each answer, stored in `brief_json` from 2026-10-30 (the 2026-10-29 round predates this field)

   A replacement model starts a **new** track with its own warm-up and its own analysis. Results are reported per track, and tracks are never pooled into one number. `--status` lists the builds seen per session so a change is visible the evening it happens.
7. *Second track (added 2026-10-30, before any outcome exists).* `openai/gpt-oss-120b` on Groq, reasoning effort `low`, stored as model `openai/gpt-oss-120b@low`. Everything else is shared with the first track:
   - the same prompt (`v1_structured`) and the same 41 stocks
   - the identical price snapshot: each stock is fetched once per evening and given to both models
   - the same evidence rule and the same warm-up end (2026-11-18, set by the filing store, not the model)
   - one shared hash chain

   Its first round is 2026-10-30. It is analysed exactly like the first track (items 1–5) and reported beside it. The comparison between tracks is the **paired** difference in per-session rank correlation over primary sessions both tracks have. That comparison is exploratory: at about 12 primary sessions a year it is underpowered for longer than either track alone. v1 has 671 gpt-oss briefs on a subset of windows, so this track has only a partial in-sample reference, which is reported as such.

**M4 — settlement** (`live/settle.py`, built before any call was due). Each evening after the round, every call whose 21-session window has closed is settled once into an append-only `outcomes` table. Settlement uses v1's `get_outcome` and `score_l3` unchanged. A test replays v1's cached briefs through M4's summariser and gets v1's published −0.0153 / t −0.65. Two details the plan didn't spell out were fixed before any outcome existed:
- a scheduled primary session the ledger doesn't have (the machine was off) is reported as **missing**, never replaced by a neighbouring session
- the secondary series' standard error is Newey-West at lag 20, matching the overlap of consecutive 21-day windows

```powershell
python -m scripts.settle                    # settle what's due, print the analysis (also runs nightly)
python -m scripts.live_predict              # today's round (after the close)
python -m scripts.live_predict --status
python -m scripts.live_predict --verify     # recompute the hash chain
streamlit run app/live_dashboard.py         # "Predictions" page: each day's calls, their evidence, chain + GitHub check
python -m scripts.export_site               # static JSON for site/ (ledger + daily candles)
python -m http.server 8700 -d site          # terminal-style ledger view at http://localhost:8700
powershell -ExecutionPolicy Bypass -File scripts\register_ledger_task.ps1   # weekdays 18:30 + 21:30
```

## The leaks this guards against

**1. `info` and `fast_info` are always *now*.** AAPL's 52-week high read today is $344.57; reconstructed as of 2025-11-03 it was **$271.40**. A model given the former is told the stock sits 21% below its high when it was sitting near it — an inverted signal, delivered as fact. Everything comes from `history()` instead.

**2. yfinance back-adjusts closes for splits that happen *after* your date.** RELIANCE closed at **₹3,020.65** on 2024-06-03; yfinance stores ₹1,510.32 because of a 1:2 split five months later. This bites far harder on NSE than in US markets — NESTLEIND split 10× (Jan 2024) *and* 2× (Aug 2025), so a snapshot dated 2024-01-02 must undo a factor of **20**.

**3. Filings published after the cutoff.** BSE timestamps are exchange filing times, authoritative to the second, so the comparison is exact. 2,650 filings (6%) fell inside the fetch window but after the cutoff and were dropped.

Trading holidays need no calendar — `history()` returns only real trading rows, so "21 trading days" is 21 rows. `^NSEI` history *is* the NSE calendar.

## The leak tests

Five are named `test_leak_1` … `test_leak_5` in `tests/test_market_data.py`; the sixth guards the disclosure cutoff in `tests/test_disclosure_selection.py`.

1. No row after the cutoff is used — pinned against a known violent next day (Reliance fell ~7% on 2024-06-04).
2. Reconstructed 52w high must *differ* from today's `info` — if they ever agree, someone wired `info` back in.
3. **Golden file** — snapshots for three fixed past dates, committed. The past does not change.
4. A post-cutoff split is undone in the reported price.
5. Two cutoffs disagree — a snapshot is a function of its date, not of when it ran.
6. No selected filing may carry a timestamp at or after the cutoff.

`tests/test_frozen_v1.py` guards the result itself: the 90 cutoffs are committed, and the report recomputed from the local cache must equal `results-final.json` exactly. It exists because the benchmark once wasn't frozen — the window list ended at *today*, so a new window appeared every 21 trading days and the report stopped reproducing. The end is now pinned (`BENCHMARK_END`, 2026-09-01).

```powershell
pytest -m "not network"   # 276 offline
pytest                    # + live leak tests
```

## What measurement falsified

**L2's semantic tier does not work.** A token-overlap heuristic was built to check whether a claim matches the filing it cites. Hand-reading every claim it flagged showed **all of them were correct**:

```
overlap 0.00  "Acquisition of edtech and tech startups expands digital portfolio"
              filing: Indiavidual Learning Private Limited ('Embibe'), a Reliance subsidiary
```

Embibe *is* edtech. Token overlap measures shared vocabulary, and a correct paraphrase routinely shares none. Its precision against real errors was zero, so `support_rate` is marked `NOT_A_METRIC` and L2 reports tier 1 only. The semantic half needs an entailment judge validated against hand labels.

**A relative-error tolerance mis-scores small numbers.** The benchmark's own first live run flagged `change_pct_1d: 0.42` against a truth of `0.4248` as a hallucination — correct 2dp rounding, but a 1.13% relative error. Percentage fields sit near zero constantly, so this would have penalised a large share of every run. A citation now also matches when it equals the truth rounded to the decimals the model wrote.

**A partial grid produced a publishable-looking table.** The first baseline run scored 1,826 of 3,690 pairs; yfinance had throttled mid-run and every exception was caught and counted, so the job "succeeded". The missingness was systematic — tickers iterate in fixed order, so throttling always dropped the same tail of the universe. Fetching and scoring are now separate steps, and `load_grid` refuses to return an incomplete grid.

## Data sources

| Source | Use | Key |
|---|---|---|
| yfinance | prices, splits, shares outstanding | none |
| BSE announcements (`bse`) | dated primary disclosures | none |
| `^NSEI` | Nifty 50 benchmark and trading calendar | none |

**Rejected, with measured reasons:**

| Candidate | Why not |
|---|---|
| Finnhub | 403 on `.NS`/`.BO` — Indian equities are paid-tier only |
| GDELT | Doc API rate-limits *cumulatively per IP*; served the spike, then refused everything |
| Groq free | 200,000 **tokens/day** → 176 briefs/day → 21 days |
| Gemini 2.5 Flash | **20 requests/day** |
| Cerebras | 402 Payment Required — no free tier |
| OpenAI | no credits on the available key |

**Chosen:** `gemini-3.1-flash-lite` — 15 req/min, **500 req/day**, ~7 days for the full run.

Every one of those limits was found by reading the error *body*. Response headers were misleading in three separate cases — Groq reported a full per-minute token bucket while refusing every call, because the real cap was daily.

## Running it

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m scripts.build_grid          # prices + forward returns (~1h, once)
python -m scripts.fetch_disclosures   # BSE filings (~25m, once)
python -m scripts.run_baselines       # the bar the model must clear
python -m scripts.run_briefs          # generation; stops cleanly at the daily cap
python -m scripts.report              # the result
```

Every stage is cached, resumable, and refuses to score incomplete data. `run_briefs` exits cleanly when the provider's daily quota is spent rather than marking untouched cells as failures — rerun the next day to continue.

**Dashboards** (read-only, both local):

```powershell
python -m app.server                        # http://localhost:8600
python -m streamlit run app/dashboard.py    # http://localhost:8501
```

**Inspect one snapshot by hand:**

```powershell
python -m app.cli RELIANCE --as-of 2024-06-03
```

## Known limitations

- **Survivorship bias.** The universe is 41 Nifty-50-class names frozen at 2019. Freezing stops the bias growing at each rebalance; it does not remove it. TATAMOTORS is excluded because it demerged in 2025 and its BSE listing no longer resolves — itself a small bias, recorded rather than hidden.
- **Verdict concentration.** Final split: 56% NEUTRAL, 36% BULLISH, **8% BEARISH**. A three-valued predictor that is more than half one value has a low ceiling — a perfectly-ordered verdict across 41 names could not exceed roughly 0.95 rank correlation, and only 284 bearish calls across 90 dates means the long-short spread is thin or undefined on many of them. This is why rank correlation is the primary metric and the spread is secondary. The measured −0.0153 is far enough from any ceiling that concentration does not explain it, but a model with more directional conviction would be a better test of the harness.
- **Confidence is useless.** The model emits only 3 or 4, never 1, 2 or 5. It is not used to weight anything.
- **Administrative filings survive the noise filter.** Investor-conference notices are genuine filings carrying no information; a model citing one passes L2 tier 1 with a vacuous claim. Another argument for the entailment judge.
- **P/E is deliberately absent.** EPS is not public until the report date, so using it earlier is a leak that looks like clean data.
- **`.BO` is unsupported** — yfinance returns nothing for it.

## Layout

```
data/       point-in-time snapshots, disclosures, universe, windows
evals/      grid, outcomes, baselines, scoring, number_check, attribution, report
llm/        prompts (versioned), brief parser, provider-agnostic client
scripts/    build_grid, fetch_disclosures, run_baselines, run_briefs, report
app/        cli, server + static dashboard, streamlit dashboard
tests/      276 offline, plus live leak tests
```
