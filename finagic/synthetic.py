"""A small fake market in the exact raw formats the real pipeline reads, for testing offline.

It plants one honest signal (companies with higher underlying profitability earn a little
more over the next year) so the backtest has something real to find, and it writes 10-Ks with
filing delays and restated comparatives so the timing and restatement leaks are exercised.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

SECTORS = ["Industrials", "Information Technology", "Health Care", "Consumer Staples", "Energy"]


def make(root: Path, n_stocks: int = 90, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    sec_dir = root / "sec"
    sec_dir.mkdir(exist_ok=True)
    months = pd.date_range("2009-01-31", "2026-09-30", freq="ME")
    T = len(months)

    rows, price_rows = [], []
    market = rng.normal(0.007, 0.04, T)
    for i in range(n_stocks):
        ticker, cik = f"SYN{i:03d}", 900000 + i
        sector = SECTORS[i % len(SECTORS)]
        added = "1990-01-01" if i % 4 else f"{rng.integers(2012, 2021)}-06-30"
        rows.append({"Symbol": ticker, "Security": f"Synthetic {i}", "GICS Sector": sector,
                     "GICS Sub-Industry": "x", "Headquarters Location": "x", "Date added": added,
                     "CIK": cik, "Founded": 2000})

        # latent profitability drifts slowly; it drives both fundamentals and future returns
        quality = np.cumsum(rng.normal(0, 0.02, T)) + rng.normal(0.10, 0.05)
        size = rng.lognormal(23, 1.0)
        rets = market * rng.uniform(0.6, 1.4) + 0.004 * (quality - 0.10) / 0.05 + rng.normal(0, 0.06, T)
        adj = 50 * np.cumprod(1 + rets)
        split = np.ones(T)
        if i == 3:
            split[months >= pd.Timestamp("2020-08-31")] = 4.0       # a 4-for-1 split
        close = adj * (1 - 0.0015) ** np.arange(T)[::-1]            # dividends make close > adj in the past
        for t in range(T):
            price_rows.append({"date": months[t].date().isoformat(), "close": close[t], "adj": adj[t],
                               "split": split[t], "ticker": ticker})

        facts = []
        prev = None
        for year in range(2009, 2026):
            t_end = np.searchsorted(months, pd.Timestamp(f"{year}-12-31"))
            q = quality[min(t_end, T - 1)]
            rev = size * (1 + 0.05 * (year - 2009)) * rng.uniform(0.9, 1.1)
            vals = {
                "Revenues": rev, "CostOfRevenue": rev * (0.6 - q), "OperatingIncomeLoss": rev * q,
                "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": rev * q * 0.9,
                "NetIncomeLoss": rev * q * 0.7, "DepreciationDepletionAndAmortization": rev * 0.05,
                "NetCashProvidedByUsedInOperatingActivities": rev * (q + 0.05),
                "Assets": rev * 1.5, "AssetsCurrent": rev * 0.5, "LiabilitiesCurrent": rev * 0.3,
                "Liabilities": rev * 0.8, "StockholdersEquity": rev * 0.7, "InventoryNet": rev * 0.1,
                "AccountsReceivableNetCurrent": rev * 0.12, "AccountsPayableCurrent": rev * 0.08,
                "CashAndCashEquivalentsAtCarryingValue": rev * 0.1, "LongTermDebtNoncurrent": rev * rng.uniform(0.1, 0.6),
            }
            filed = f"{year + 1}-02-{rng.integers(15, 28)}"
            accn = f"{cik}-{year}"
            for c, v in vals.items():
                rec = {"c": c, "end": f"{year}-12-31", "val": float(v), "accn": accn, "filed": filed, "form": "10-K"}
                if c in ("Revenues", "CostOfRevenue", "OperatingIncomeLoss", "NetIncomeLoss",
                         "DepreciationDepletionAndAmortization", "NetCashProvidedByUsedInOperatingActivities") \
                        or c.startswith("IncomeLoss"):
                    rec["start"] = f"{year}-01-01"
                facts.append(rec)
            shares = size * 0.7 / 20 * (4.0 if i == 3 and year >= 2020 else 1.0)  # raw count jumps at the split
            facts.append({"c": "EntityCommonStockSharesOutstanding", "end": f"{year + 1}-01-31", "val": float(shares),
                          "accn": accn, "filed": filed, "form": "10-K"})
            # prior-year comparative, restated by a few percent
            if prev is not None:
                for c, v in prev.items():
                    rec = {"c": c, "end": f"{year - 1}-12-31", "val": float(v * rng.uniform(0.95, 1.05)),
                           "accn": accn, "filed": filed, "form": "10-K"}
                    if c in ("Revenues", "NetIncomeLoss", "OperatingIncomeLoss"):
                        rec["start"] = f"{year - 1}-01-01"
                        facts.append(rec)
            prev = {k: vals[k] for k in ("Revenues", "NetIncomeLoss", "OperatingIncomeLoss")}
        with gzip.open(sec_dir / f"{cik}.json.gz", "wt") as f:
            json.dump(facts, f)

    pd.DataFrame(rows).to_csv(root / "constituents.csv", index=False)
    pd.DataFrame(price_rows).to_csv(root / "prices.csv", index=False)
    return {"constituents": root / "constituents.csv", "sec_dir": sec_dir, "prices": root / "prices.csv"}
