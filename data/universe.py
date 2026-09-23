"""The benchmark universe: NSE large caps, fixed once, with BSE scrip codes.

**The universe is frozen, not live.** Picking today's Nifty 50 would select
companies that survived and thrived — a winners' list. Freezing a list chosen
at the benchmark's start date does not remove survivorship bias (these are
still names that existed and were large in 2019), but it stops the bias from
growing every time the index rebalances. The residual bias is stated in the
README rather than pretended away.

**Every scrip code below was resolved from BSE and round-trip verified** by
asking BSE what each code maps back to. A wrong code does not error — it
silently returns another company's filings — so the codes are committed here
rather than looked up at runtime, where a flaky lookup could substitute one
company for another mid-run.

**TATAMOTORS is deliberately absent.** Tata Motors demerged in 2025 and the old
BSE listing no longer resolves. Guessing a replacement code would be inventing
data; dropping it is itself a small survivorship bias (a company excluded
*because* it restructured), which is why it is written down here.

`gdelt_query` is retained but unused: the GDELT Doc API rate-limits too hard
for a bulk historical fetch (see the plan). It is kept for a possible BigQuery
path, where the same query strings would apply.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Company:
    symbol: str          # NSE symbol, without the .NS suffix
    name: str            # display name
    scrip_code: str      # BSE scrip code, round-trip verified
    gdelt_query: str     # retained for a possible BigQuery path; unused today

    @property
    def ticker(self) -> str:
        return f"{self.symbol}.NS"


def _q(name: str) -> str:
    return f'"{name}" sourcecountry:india'


UNIVERSE: tuple[Company, ...] = (
    Company("RELIANCE",   "Reliance Industries",             "500325", _q("Reliance Industries")),
    Company("TCS",        "Tata Consultancy Services",       "532540", _q("Tata Consultancy Services")),
    Company("HDFCBANK",   "HDFC Bank",                       "500180", _q("HDFC Bank")),
    Company("INFY",       "Infosys",                         "500209", _q("Infosys")),
    Company("ICICIBANK",  "ICICI Bank",                      "532174", _q("ICICI Bank")),
    Company("HINDUNILVR", "Hindustan Unilever",              "500696", _q("Hindustan Unilever")),
    Company("ITC",        "ITC Limited",                     "500875", _q("ITC Limited")),
    Company("SBIN",       "State Bank of India",             "500112", _q("State Bank of India")),
    Company("BHARTIARTL", "Bharti Airtel",                   "532454", _q("Bharti Airtel")),
    Company("KOTAKBANK",  "Kotak Mahindra Bank",             "500247", _q("Kotak Mahindra Bank")),
    Company("LT",         "Larsen & Toubro",                 "500510", _q("Larsen and Toubro")),
    Company("AXISBANK",   "Axis Bank",                       "532215", _q("Axis Bank")),
    Company("ASIANPAINT", "Asian Paints",                    "500820", _q("Asian Paints")),
    Company("MARUTI",     "Maruti Suzuki",                   "532500", _q("Maruti Suzuki")),
    Company("BAJFINANCE", "Bajaj Finance",                   "500034", _q("Bajaj Finance")),
    Company("TITAN",      "Titan Company",                   "500114", _q("Titan Company")),
    Company("SUNPHARMA",  "Sun Pharmaceutical",              "524715", _q("Sun Pharmaceutical")),
    Company("ULTRACEMCO", "UltraTech Cement",                "532538", _q("UltraTech Cement")),
    Company("WIPRO",      "Wipro",                           "507685", _q("Wipro")),
    Company("NESTLEIND",  "Nestle India",                    "500790", _q("Nestle India")),
    Company("TATASTEEL",  "Tata Steel",                      "500470", _q("Tata Steel")),
    Company("POWERGRID",  "Power Grid Corporation",          "532898", _q("Power Grid Corporation")),
    Company("NTPC",       "NTPC Limited",                    "532555", _q("NTPC Limited")),
    Company("ONGC",       "Oil and Natural Gas Corporation", "500312", _q("Oil and Natural Gas Corporation")),
    Company("M&M",        "Mahindra & Mahindra",             "500520", _q("Mahindra and Mahindra")),
    Company("HCLTECH",    "HCL Technologies",                "532281", _q("HCL Technologies")),
    Company("JSWSTEEL",   "JSW Steel",                       "500228", _q("JSW Steel")),
    Company("INDUSINDBK", "IndusInd Bank",                   "532187", _q("IndusInd Bank")),
    Company("GRASIM",     "Grasim Industries",               "500300", _q("Grasim Industries")),
    Company("BAJAJFINSV", "Bajaj Finserv",                   "532978", _q("Bajaj Finserv")),
    Company("DRREDDY",    "Dr Reddy's Laboratories",         "500124", _q("Dr Reddy's Laboratories")),
    Company("CIPLA",      "Cipla",                           "500087", _q("Cipla")),
    Company("BRITANNIA",  "Britannia Industries",            "500825", _q("Britannia Industries")),
    Company("EICHERMOT",  "Eicher Motors",                   "505200", _q("Eicher Motors")),
    Company("COALINDIA",  "Coal India",                      "533278", _q("Coal India")),
    Company("HEROMOTOCO", "Hero MotoCorp",                   "500182", _q("Hero MotoCorp")),
    Company("BPCL",       "Bharat Petroleum",                "500547", _q("Bharat Petroleum")),
    Company("TECHM",      "Tech Mahindra",                   "532755", _q("Tech Mahindra")),
    Company("ADANIPORTS", "Adani Ports",                     "532921", _q("Adani Ports")),
    Company("HINDALCO",   "Hindalco Industries",             "500440", _q("Hindalco Industries")),
    Company("UPL",        "UPL Limited",                     "512070", _q("UPL Limited")),
)

BY_SYMBOL: dict[str, Company] = {c.symbol: c for c in UNIVERSE}
BY_TICKER: dict[str, Company] = {c.ticker: c for c in UNIVERSE}


def get(symbol_or_ticker: str) -> Company:
    """Look up a company by 'RELIANCE' or 'RELIANCE.NS'."""
    key = symbol_or_ticker.strip().upper()
    if key in BY_TICKER:
        return BY_TICKER[key]
    if key in BY_SYMBOL:
        return BY_SYMBOL[key]
    raise KeyError(f"{symbol_or_ticker!r} is not in the benchmark universe.")


def tickers() -> list[str]:
    return [c.ticker for c in UNIVERSE]
