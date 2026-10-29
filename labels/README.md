# M2 label sheets

Each rule version has its own folder (`rules-v1/`, `rules-v2/`). Each folder holds a `label_sheet.csv` of 50 NSE filings that arrived **after** that version's rules were frozen. Fill in the `label` column for each one with a single category from the table below, typed exactly as shown. The sheet deliberately leaves out the rules' answer. Label from the description, and open the `url` when the description is too vague to tell.

When you're done, run `python -m scripts.label_filings score`. If the labels came from an AI tool, add `--sheet <file> --labeller ai_tool`.

## One question decides it

> **What new information does this filing give someone holding the stock?**

Label the filing by its new information, not by its form. A newspaper reprint of results adds nothing, because the results were already filed separately, so it is `routine_compliance`. A press release whose headline is the quarter's profit is `results`.

## Categories

| label | use it for |
|---|---|
| `results` | financial results, or a press release or update whose subject is the period's numbers |
| `board_meeting` | notice of an upcoming board meeting, or an "outcome" that names no decision |
| `investor_comms` | analyst or investor meets, earnings-call notices and transcripts, investor presentations, KPIs |
| `corporate_action` | dividend, record date, bonus, split, buyback, rights issue, name change |
| `capital_raise` | allotment, issue of securities, fund-raising approvals, ESOP grants and exercises |
| `deal` | acquisition, disposal, merger or demerger, scheme, open offer, JV, subsidiary set-up, material agreement |
| `business_update` | orders and contracts, capacity, launches, plant or operations news, any other real company news |
| `management_change` | appointment, resignation, cessation, retirement or death of a director, KMP or senior manager |
| `credit_rating` | a new, revised or reaffirmed rating |
| `legal_regulatory` | regulator or exchange action, fines, litigation, fraud, default, cautionary letters |
| `debt_and_proceeds` | NCD interest or redemption, security cover, guarantees, statements on how issue money was used |
| `shareholder_meeting` | AGM, EGM or postal-ballot notice, proceedings, voting results |
| `routine_compliance` | newspaper reprints, trading window, insider-code and certificate filings, committee changes |
| `fund_nav` | a mutual fund or ETF NAV (these shouldn't appear in the sample) |
| `other` | nothing above fits, **or the filing says too little to tell**, e.g. "X has informed the Exchange about General Updates" with an empty PDF title |

## Tie-breaks

- **One filing, several items:** pick the one that would matter most to a holder. A board outcome with results and a dividend is `results`.
- **Too vague to tell:** if the description is vague, look at the PDF. If the PDF is unclear too, use `other`. Don't guess from the company name.
- **Use `notes`** whenever you hesitated. The rows you hesitated on are the useful ones when reading the disagreements.

## Who labelled what

| file | labeller | status |
|---|---|---|
| `rules-v1/labels_ai_tool.csv` | an AI tool (2026-10-29 15:34 IST) | scored: 36/50 agreement (`rules-v1/score_ai_tool.json`) |
| `rules-v1/label_sheet.csv` | — | left blank; nobody hand-labelled v1 |
| `rules-v2/label_sheet.csv` | to be drawn after 16:00 IST on 2026-10-29 | pending |

Every score file records who labelled the sheet (`"labeller"`). Agreement with AI-tool labels is reported as exactly that, never as accuracy against hand labels.

## Rules for the sample itself

- The sample is drawn **once** (fixed seed, `sample.json`). Don't redraw it to get a better set.
- The first score is the result. If the rules are changed after you've seen the disagreements, the next score is in-sample, and `score` will say so.
