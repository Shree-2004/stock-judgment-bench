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
| **L3** | Was the call right? | forward 21-trading-day return | pending full run |

**L3 is scored cross-sectionally**: within each date, do the model's bullish names outperform its bearish ones? That is market-neutral by construction — which is why `always_bullish`, the strategy that would look brilliant on raw returns, is *unscoreable* here rather than strong.

One date is one observation. 3,690 briefs do not become 3,690 degrees of freedom; they become 90.

## Results so far

**Baselines — final, all 90 dates, complete grid:**

| strategy | dates | rank corr | t | spread % | hit |
|---|---|---|---|---|---|
| always_bullish | 90 | — | — | — | — |
| random | 90 | +0.012 | 0.72 | +0.21 | 53.3% |
| momentum | 90 | −0.047 | −1.88 | −0.92 | 38.2% |
| reversal | 90 | +0.047 | +1.88 | +0.92 | 61.8% |

Buying last month's *losers* beat buying last month's winners by ~0.9%/month. Suggestive, not significant (p ≈ 0.06). Momentum and reversal are mirrors, so that is one finding, not two.

**Model (`gemini-3.1-flash-lite`), 428 of 3,690 briefs:**

- **L1: zero wrong figures, zero invented labels** across 428 briefs
- **L2 tier 1: zero fabricated citations, zero uncited claims**
- Verdicts: 57% NEUTRAL, 34% BULLISH, 9% BEARISH
- L3: pending

**Pre-registered prediction**, recorded before any briefs existed: the model's verdicts track one-month momentum, so if it is merely re-reading momentum it should land near **−0.047**, not near zero.

## Status

| Step | What | State |
|---|---|---|
| 1 | Point-in-time data layer + leak tests | done |
| 2 | BSE disclosure cache + selection rule | done |
| 3 | Brief generator, L1 grounding, Langfuse tracing | done |
| 4 | Grid, baselines, L3 cross-sectional scoring | done |
| 5 | L2 attribution — tier 1 done, tier 2 falsified | partial |
| 6 | Memorisation control, prose-vs-structured ablation | not started |
| — | Full 3,690-brief run | 428 done, ~7 days at 500/day |

## The leaks this guards against

**1. `info` and `fast_info` are always *now*.** AAPL's 52-week high read today is $344.57; reconstructed as of 2025-11-03 it was **$271.40**. A model given the former is told the stock sits 21% below its high when it was sitting near it — an inverted signal, delivered as fact. Everything comes from `history()` instead.

**2. yfinance back-adjusts closes for splits that happen *after* your date.** RELIANCE closed at **₹3,020.65** on 2024-06-03; yfinance stores ₹1,510.32 because of a 1:2 split five months later. This bites far harder on NSE than in US markets — NESTLEIND split 10× (Jan 2024) *and* 2× (Aug 2025), so a snapshot dated 2024-01-02 must undo a factor of **20**.

**3. Filings published after the cutoff.** BSE timestamps are exchange filing times, authoritative to the second, so the comparison is exact. 2,650 filings (6%) fell inside the fetch window but after the cutoff and were dropped.

Trading holidays need no calendar — `history()` returns only real trading rows, so "21 trading days" is 21 rows. `^NSEI` history *is* the NSE calendar.

## The leak tests

1. No row after the cutoff is used — pinned against a known violent next day (Reliance fell ~7% on 2024-06-04).
2. Reconstructed 52w high must *differ* from today's `info` — if they ever agree, someone wired `info` back in.
3. **Golden file** — snapshots for three fixed past dates, committed. The past does not change.
4. A post-cutoff split is undone in the reported price.
5. Two cutoffs disagree — a snapshot is a function of its date, not of when it ran.
6. No selected filing may carry a timestamp at or after the cutoff.

```powershell
pytest -m "not network"   # 169 offline
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
- **Verdict concentration.** At 57% NEUTRAL, the predictor has few distinct values. If that holds, L3 may be underpowered for the same structural reason `always_bullish` is unscoreable, and the honest finding becomes "too concentrated to test".
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
tests/      169 offline, plus live leak tests
```
