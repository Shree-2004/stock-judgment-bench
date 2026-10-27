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

### The pre-registered prediction was falsified

Recorded before any brief existed: an early 16-ticker probe showed verdicts tracking one-month momentum, so if the model were merely re-reading momentum it should land near **−0.047**.

Final: **−0.0153** — a third of that, and statistically zero. The expected direction (no edge) held; the mechanism named for it did not. The probe was too small to generalise.

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
| — | **Full 3,690-brief run** | **done** |

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
pytest -m "not network"   # 173 offline
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
tests/      173 offline, plus live leak tests
```
