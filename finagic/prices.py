"""Monthly prices from Yahoo Finance (via yfinance), three series per month-end:

  close  close price, adjusted for splits only (Yahoo's "Close")
  adj    adjusted for splits and dividends -> total returns
  split  cumulative product of split ratios up to that month

Market cap at month t from a 10-K share count S_f taken at filing month f:
  mcap(t) = close(t) * S_f * split(last) / split(f)
because "close" is already restated into today's share units.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def download(tickers: list[str], start: str, out_path: Path, chunk: int = 40) -> pd.DataFrame:
    import yfinance as yf  # local-only dependency

    frames = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        syms = [yahoo_symbol(t) for t in batch]
        print(f"  prices {i + 1}-{i + len(batch)} of {len(tickers)}")
        for attempt in range(4):
            try:
                df = yf.download(syms, start=start, auto_adjust=False, actions=True, group_by="ticker",
                                 progress=False, threads=True)
                break
            except Exception as e:  # Yahoo rate limits are common; back off and retry
                print(f"    retrying after: {e}")
                time.sleep(20 * (attempt + 1))
        else:
            continue
        for t, sym in zip(batch, syms):
            if sym not in df.columns.get_level_values(0):
                continue
            d = df[sym].dropna(subset=["Close"])
            if d.empty:
                continue
            splits = d["Stock Splits"].replace(0, 1.0).fillna(1.0)
            daily = pd.DataFrame({"close": d["Close"], "adj": d["Adj Close"], "split": splits.cumprod()})
            monthly = daily.resample("ME").last()
            monthly["ticker"] = t
            frames.append(monthly)
        time.sleep(1.0)
    out = pd.concat(frames).reset_index(names="date")
    out.to_csv(out_path, index=False)
    return out


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["yyyymm"] = df["date"].dt.year * 100 + df["date"].dt.month
    return df


def to_matrices(df: pd.DataFrame, tickers: list[str], months: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for col in ("close", "adj", "split"):
        wide = df.pivot_table(index="yyyymm", columns="ticker", values=col, aggfunc="last")
        wide = wide.reindex(index=months, columns=tickers)
        out[col] = wide.to_numpy(dtype=np.float64)
    return out
