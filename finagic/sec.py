"""Fetch annual (10-K) fundamentals from SEC EDGAR's free XBRL API and keep their filing dates.

Every SEC fact carries the accession number and the date it was filed, which is what makes
a point-in-time backtest possible: a number can only be used after the day it was filed.

Docs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
SEC asks for a descriptive User-Agent with contact details and at most 10 requests/second.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from datetime import date
from pathlib import Path

import requests

CONSTITUENTS_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Field -> XBRL concepts to try in order. Flows are annual (duration ~1 year); stocks are instants.
FLOW_CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"],
    "gross_profit": ["GrossProfit"],
    "op_income": ["OperatingIncomeLoss"],
    "pretax": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet", "Depreciation"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
}
STOCK_CONCEPTS = {
    "assets": ["Assets"],
    "cur_assets": ["AssetsCurrent"],
    "cur_liab": ["LiabilitiesCurrent"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "inventory": ["InventoryNet"],
    "ar": ["AccountsReceivableNetCurrent"],
    "ap": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "ltd": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "std": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
}
SHARES_CONCEPTS = [("dei", "EntityCommonStockSharesOutstanding"),
                   ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")]
FIELDS = list(FLOW_CONCEPTS) + list(STOCK_CONCEPTS) + ["shares"]


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _is_annual(f: dict) -> bool:
    return "start" in f and 330 <= _days(f["start"], f["end"]) <= 400


# ---------------------------------------------------------------- download
def user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        raise SystemExit('Set SEC_USER_AGENT to "Your Name your@email.com" (SEC requires contact details).')
    return ua


def download_constituents(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        r = requests.get(CONSTITUENTS_URL, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def compact_facts(raw: dict) -> list[dict]:
    """Keep only 10-K facts for the concepts we use. ~100x smaller than the raw file."""
    wanted = {("us-gaap", c) for cs in list(FLOW_CONCEPTS.values()) + list(STOCK_CONCEPTS.values()) for c in cs}
    wanted |= set(SHARES_CONCEPTS)
    out = []
    for tax, concepts in raw.get("facts", {}).items():
        for concept, body in concepts.items():
            if (tax, concept) not in wanted:
                continue
            for unit, facts in body.get("units", {}).items():
                if unit not in ("USD", "shares"):
                    continue
                for f in facts:
                    if not str(f.get("form", "")).startswith("10-K"):
                        continue
                    rec = {"c": concept, "end": f["end"], "val": f["val"], "accn": f["accn"],
                           "filed": f["filed"], "form": f["form"]}
                    if "start" in f:
                        rec["start"] = f["start"]
                    out.append(rec)
    return out


def fetch_company(cik: int, out_dir: Path, session: requests.Session) -> Path | None:
    out = out_dir / f"{cik}.json.gz"
    if out.exists():
        return out
    for attempt in range(4):
        r = session.get(FACTS_URL.format(cik=cik), timeout=60)
        if r.status_code == 200:
            facts = compact_facts(r.json())
            with gzip.open(out, "wt") as f:
                json.dump(facts, f)
            return out
        if r.status_code == 404:
            return None
        time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------- parse into filings
def filings_from_facts(facts: list[dict]) -> list[dict]:
    """One record per 10-K: {accn, filed, period_end, first: {...}, restated: {...}}.

    first     values exactly as they appeared in THIS 10-K (what the market saw on `filed`).
    restated  for the same fiscal period, the value from the LATEST 10-K that reported it
              (prior-year comparatives are often restated). Using these is a leak.
    """
    index: dict[tuple[str, str], list[dict]] = {}
    by_accn: dict[str, list[dict]] = {}
    for f in facts:
        index.setdefault((f["c"], f["end"]), []).append(f)
        by_accn.setdefault(f["accn"], []).append(f)

    anchor = {c for k in ("revenue", "net_income", "ocf") for c in FLOW_CONCEPTS[k]}
    filings = []
    for accn, fs in by_accn.items():
        ends = [f["end"] for f in fs if f["c"] in anchor and _is_annual(f)]
        if not ends:
            continue
        period_end = max(ends)
        filings.append({
            "accn": accn,
            "filed": min(f["filed"] for f in fs),
            "period_end": period_end,
            "first": _values_for(index, period_end, accn, fs),
            "restated": _values_for(index, period_end, None, fs),
        })
    filings.sort(key=lambda x: (x["filed"], x["period_end"]))
    # keep only the first 10-K per period end (a later 10-K/A is not news on its own here)
    seen, keep = set(), []
    for fl in filings:
        if fl["period_end"] in seen:
            continue
        seen.add(fl["period_end"])
        keep.append(fl)
    return keep


def _pick(index: dict, concepts: list[str], period_end: str, flow: bool, accn: str | None) -> float | None:
    for c in concepts:
        cands = [f for f in index.get((c, period_end), [])
                 if (_is_annual(f) if flow else "start" not in f) and (accn is None or f["accn"] == accn)]
        if cands:
            # accn given -> the value in that filing; accn None -> latest filing that reported it
            return float(max(cands, key=lambda f: f["filed"])["val"])
    return None


def _values_for(index: dict, period_end: str, accn: str | None, accn_facts: list[dict]) -> dict:
    vals = {k: _pick(index, cs, period_end, True, accn) for k, cs in FLOW_CONCEPTS.items()}
    vals |= {k: _pick(index, cs, period_end, False, accn) for k, cs in STOCK_CONCEPTS.items()}
    # Shares come from the cover page of the 10-K itself (dated shortly before filing).
    dei = [f for f in accn_facts if f["c"] == "EntityCommonStockSharesOutstanding"]
    if dei:
        last = max(f["end"] for f in dei)
        vals["shares"] = float(sum(f["val"] for f in dei if f["end"] == last))  # sums share classes
    else:
        vals["shares"] = _pick(index, ["WeightedAverageNumberOfDilutedSharesOutstanding"], period_end, True, accn)
    return vals
