"""Turn raw filings + prices into the scoring panel, once honestly and once with leaks.

The four leaks (each is a shortcut people really take when building backtests):

  L1 timing       a 10-K's numbers are usable from the fiscal year END, not the day it was
                  FILED (typically 4-10 weeks later).
  L2 restatements values come from the latest 10-K that reported the period, so later
                  corrections are visible in the past.
  L3 price        valuation multiples use the price at the END of the 12-month return window.
                  This is what the 2021 Finagic backtest effectively did: it scored stocks with
                  the data available on the day it ran, then measured returns over the past.
  L4 universe     every stock that is in the S&P 500 today is investable in every past month,
                  including companies that only joined the index after they had grown.

Not fixed in either environment (and said so in the guide): companies that LEFT the index
are missing from both, because the free constituent list only covers current members.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import prices as px
from .metrics import KEYS
from .sec import FIELDS, filings_from_facts

EXCLUDED_SECTORS = {"Financials", "Real Estate"}  # balance-sheet metrics don't apply to banks/REITs
MIN_MCAP = 300e6
MIN_METRICS = 10
PEERS_K = 10
MAX_STALENESS_MONTHS = 18
FIRST_MONTH = 201001


@dataclass(frozen=True)
class Leaks:
    timing: bool = False
    restated: bool = False
    future_price: bool = False
    survivors: bool = False

    @property
    def name(self) -> str:
        on = [k for k, v in asdict(self).items() if v]
        return "clean" if not on else "leaky" if len(on) == 4 else "+".join(on)


CLEAN = Leaks()
LEAKY = Leaks(True, True, True, True)


def yyyymm(d: str) -> int:
    return int(d[:4]) * 100 + int(d[5:7])


def months_between(a: int, b: int) -> int:
    return (b // 100 - a // 100) * 12 + (b % 100 - a % 100)


def shift_months(m: int, k: int) -> int:
    y, mo = divmod(m // 100 * 12 + (m % 100 - 1) + k, 12)
    return y * 100 + mo + 1


# ---------------------------------------------------------------- inputs
def load_universe(constituents_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(constituents_csv)
    df = df[~df["GICS Sector"].isin(EXCLUDED_SECTORS)].copy()
    df["added"] = pd.to_datetime(df["Date added"], errors="coerce")
    df["added_yyyymm"] = (df["added"].dt.year * 100 + df["added"].dt.month).fillna(190001).astype(int)
    return df.rename(columns={"Symbol": "ticker", "GICS Sector": "sector", "CIK": "cik"})[
        ["ticker", "cik", "sector", "added_yyyymm"]].reset_index(drop=True)


def load_filings(sec_dir: Path, cik: int) -> list[dict]:
    p = sec_dir / f"{int(cik)}.json.gz"
    if not p.exists():
        return []
    with gzip.open(p, "rt") as f:
        return filings_from_facts(json.load(f))


# ---------------------------------------------------------------- fundamentals -> metrics
def fundamentals_matrix(filings_by_stock: list[list[dict]], months: np.ndarray, leaks: Leaks):
    """F[t, s, field] from the latest usable 10-K, plus the filing month (for split adjustment)."""
    T, S = len(months), len(filings_by_stock)
    F = np.full((T, S, len(FIELDS)), np.nan)
    filed_m = np.zeros((T, S), dtype=np.int64)
    view = "restated" if leaks.restated else "first"
    for s, fls in enumerate(filings_by_stock):
        if not fls:
            continue
        avail = np.array([yyyymm(f["period_end"] if leaks.timing else f["filed"]) for f in fls])
        order = np.argsort(avail, kind="stable")
        avail, fls = avail[order], [fls[i] for i in order]
        pend = np.array([yyyymm(f["period_end"]) for f in fls])
        vals = np.array([[np.nan if f[view].get(k) is None else f[view][k] for k in FIELDS] for f in fls])
        fm = np.array([yyyymm(f["filed"]) for f in fls])
        pos = np.searchsorted(avail, months, side="right") - 1
        for t, i in enumerate(pos):
            if i < 0 or months_between(pend[i], months[t]) > MAX_STALENESS_MONTHS:
                continue
            F[t, s] = vals[i]
            filed_m[t, s] = fm[i]
    return F, filed_m


def compute_metrics(F: np.ndarray, mcap: np.ndarray, shares: np.ndarray) -> np.ndarray:
    g = {k: F[..., i] for i, k in enumerate(FIELDS)}
    with np.errstate(divide="ignore", invalid="ignore"):
        rev, cogs = g["revenue"], g["cogs"]
        gp = np.where(np.isnan(g["gross_profit"]), rev - cogs, g["gross_profit"])
        debt = np.nan_to_num(g["ltd"]) + np.nan_to_num(g["std"])
        debt = np.where(np.isnan(g["ltd"]) & np.isnan(g["std"]), np.nan, debt)
        ev = mcap + np.nan_to_num(debt) - np.nan_to_num(g["cash"])
        ebitda = g["op_income"] + g["da"]
        eq = np.where(g["equity"] > 0, g["equity"], np.nan)
        pos = lambda x: np.where(x > 0, x, np.nan)  # noqa: E731
        dso = g["ar"] / pos(rev) * 365
        dio = g["inventory"] / pos(cogs) * 365
        dpo = g["ap"] / pos(cogs) * 365

        def _tg(x, h=12):  # trailing 1-yr growth; needs a positive base, else NaN
            out = np.full_like(x, np.nan)
            out[h:] = x[h:] / np.where(x[:-h] > 0, x[:-h], np.nan) - 1
            return out

        m = {
            "ev_ebitda": ev / pos(ebitda),
            "ps": mcap / pos(rev),
            "pcf": mcap / pos(g["ocf"]),
            "pb": mcap / eq,
            "bvps": g["equity"] / pos(shares),
            "roa": g["net_income"] / pos(g["assets"]),
            "roe": g["net_income"] / eq,
            "roce": g["op_income"] / pos(g["assets"] - g["cur_liab"]),
            "gross_margin": gp / pos(rev),
            "op_margin": g["op_income"] / pos(rev),
            "pretax_margin": g["pretax"] / pos(rev),
            "net_margin": g["net_income"] / pos(rev),
            "current_ratio": g["cur_assets"] / pos(g["cur_liab"]),
            "quick_ratio": (g["cur_assets"] - np.nan_to_num(g["inventory"])) / pos(g["cur_liab"]),
            "ccc": dso + np.nan_to_num(dio) - dpo,
            "dpo": dpo,
            "dio": dio,
            "dso": dso,
            "de": debt / eq,
            "cf_coverage": g["ocf"] / pos(debt),
            "ltd_cap": g["ltd"] / pos(g["ltd"] + eq),
            "tg_rev": _tg(rev),
            "tg_gp": _tg(gp),
            "tg_ocf": _tg(g["ocf"]),
        }
    X = np.stack([m[k] for k in KEYS], axis=-1)
    X[~np.isfinite(X)] = np.nan
    return X


# ---------------------------------------------------------------- Finagic points
def finagic_points(X: np.ndarray, mcap: np.ndarray, valid: np.ndarray, sector: np.ndarray):
    """Min-max points 0-10 against each stock's peers (same sector, nearest market caps) and
    the sector median. P: missing metric -> NaN. P0: missing company value treated as 0 (v0)."""
    T, S, M = X.shape
    P = np.full((T, S, M), np.nan, dtype=np.float32)
    P0 = np.full((T, S, M), np.nan, dtype=np.float32)
    logcap = np.log(np.where(mcap > 0, mcap, np.nan))
    for t in range(T):
        for g in np.unique(sector):
            members = np.flatnonzero(valid[t] & (sector == g))
            if len(members) < 2:
                continue
            Xg = X[t, members]
            med = np.nanmedian(Xg, axis=0) if len(members) else np.full(M, np.nan)
            lc = logcap[t, members]
            for a, s in enumerate(members):
                dist = np.abs(lc - lc[a])
                dist[a] = np.inf
                peers = np.argsort(dist)[:PEERS_K]
                ref = np.vstack([Xg[peers], med[None, :]])
                x = Xg[a]
                for out, xs in ((P, x), (P0, np.nan_to_num(x, nan=0.0))):
                    allv = np.vstack([ref, xs[None, :]])
                    with np.errstate(all="ignore"):
                        lo, hi = np.nanmin(allv, axis=0), np.nanmax(allv, axis=0)
                        pts = np.where(hi > lo, 10 * (xs - lo) / (hi - lo), 5.0)
                    pts[np.isnan(xs)] = np.nan
                    out[t, s] = pts
    return P, P0


# ---------------------------------------------------------------- one environment
def build_env(uni: pd.DataFrame, filings: list[list[dict]], prc: dict, months: np.ndarray, leaks: Leaks):
    T, S = len(months), len(uni)
    F, filed_m = fundamentals_matrix(filings, months, leaks)
    close, split = prc["close"], prc["split"]
    split_last = np.array([np.nanmax(split[:, s]) if np.isfinite(split[:, s]).any() else np.nan for s in range(S)])
    m_index = {m: i for i, m in enumerate(months)}
    fi = np.vectorize(lambda m: m_index.get(m, -1))(filed_m)
    split_at_f = np.where(fi >= 0, np.take_along_axis(split, np.clip(fi, 0, T - 1), axis=0), np.nan)
    shares = F[..., FIELDS.index("shares")] * split_last[None, :] / split_at_f   # in today's units

    price = close.copy()
    if leaks.future_price:
        price = np.full_like(close, np.nan)
        price[:-12] = close[12:]
    mcap = price * shares
    X = compute_metrics(F, mcap, shares)

    in_index = np.ones((T, S), bool) if leaks.survivors else (uni["added_yyyymm"].to_numpy()[None, :] <= months[:, None])
    enough = (~np.isnan(X)).sum(axis=2) >= MIN_METRICS
    valid = in_index & enough & (mcap >= MIN_MCAP)
    sector_ids = pd.Categorical(uni["sector"]).codes.astype(np.int8)
    P, P0 = finagic_points(X, mcap, valid, sector_ids)
    return {"P": P, "P0": P0, "valid": valid, "X": X, "mcap": mcap,
            "rev": F[..., FIELDS.index("revenue")]}


def forward_returns(adj: np.ndarray, h: int = 12) -> np.ndarray:
    R = np.full_like(adj, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        R[:-h] = adj[h:] / adj[:-h] - 1
    return R


def forward_growth(x: np.ndarray, h: int = 12) -> np.ndarray:
    """Realized growth over the next h months: x[t+h]/x[t]-1 (needs a positive base)."""
    G = np.full_like(x, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        G[:-h] = x[h:] / np.where(x[:-h] > 0, x[:-h], np.nan) - 1
    return G


def sector_demean(G: np.ndarray, sector: np.ndarray, valid: np.ndarray, min_n: int = 6) -> np.ndarray:
    """Subtract each month's within-sector median over valid members; NaN elsewhere.
    This strips sector tailwinds so the target isolates company-specific growth."""
    out = np.full_like(G, np.nan)
    for t in range(G.shape[0]):
        for g in np.unique(sector):
            m = valid[t] & (sector == g) & np.isfinite(G[t])
            if int(m.sum()) >= min_n:
                out[t, m] = G[t, m] - np.median(G[t, m])
    return out


def trailing_growth(x: np.ndarray, h: int = 12) -> np.ndarray:
    """Realized growth over the PREVIOUS h months: x[t]/x[t-h]-1 (needs a positive base)."""
    TG = np.full_like(x, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        TG[h:] = x[h:] / np.where(x[:-h] > 0, x[:-h], np.nan) - 1
    return TG


def residualize(y: np.ndarray, x_list: list[np.ndarray], valid: np.ndarray, min_n: int = 20) -> np.ndarray:
    """Per month: OLS residual of y on [1, *x_list] across valid names, i.e. remove the
    linear part explained by the x features. Here it strips the momentum factor
    (trailing growth) from forward growth, leaving the harder-to-predict residual."""
    out = np.full_like(y, np.nan)
    for t in range(y.shape[0]):
        finite_x = np.all(np.stack([np.isfinite(x[t]) for x in x_list]), axis=0)
        ok = valid[t] & np.isfinite(y[t]) & finite_x
        n = int(ok.sum())
        if n < min_n:
            continue
        Y = y[t, ok].astype(float)
        X = np.column_stack([np.ones(n)] + [x[t, ok].astype(float) for x in x_list])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        out[t, ok] = Y - X @ beta
    return out


def build_all(constituents_csv: Path, sec_dir: Path, prices_csv: Path, out_npz: Path,
              ablation_npz: Path | None = None) -> dict:
    uni = load_universe(constituents_csv)
    pr = px.load(prices_csv)
    last = int(pr["yyyymm"].max())
    months = []
    m = FIRST_MONTH
    while m <= last:
        months.append(m)
        m = shift_months(m, 1)
    months = np.array(months, dtype=np.int64)
    tickers = uni["ticker"].tolist()
    prc = px.to_matrices(pr, tickers, months)
    print(f"universe: {len(tickers)} stocks, months {months[0]}-{months[-1]}")
    filings = [load_filings(sec_dir, c) for c in uni["cik"]]
    print(f"filings loaded for {sum(1 for f in filings if f)} / {len(filings)} companies")

    R = forward_returns(prc["adj"])
    has_fwd = np.zeros(len(months), bool)
    has_fwd[:-12] = True
    envs = {"clean": build_env(uni, filings, prc, months, CLEAN),
            "leaky": build_env(uni, filings, prc, months, LEAKY)}

    # Growth target: next-12m revenue growth from point-in-time (clean) revenue,
    # sector-demeaned. Realized truth, used only as the reward label.
    sector_ids = pd.Categorical(uni["sector"]).codes.astype(np.int8)
    F_clean, _ = fundamentals_matrix(filings, months, CLEAN)
    rev_clean = F_clean[..., FIELDS.index("revenue")]
    gp_clean = np.where(np.isnan(F_clean[..., FIELDS.index("gross_profit")]),
                        rev_clean - F_clean[..., FIELDS.index("cogs")],
                        F_clean[..., FIELDS.index("gross_profit")])
    ocf_clean = F_clean[..., FIELDS.index("ocf")]
    # Target = next-12m revenue growth with the MOMENTUM factor residualized out
    # (trailing revenue/gross-profit/cash-flow growth), then sector-demeaned. What
    # remains is growth NOT explained by recent momentum -> restores RL headroom.
    G_fwd = forward_growth(rev_clean)
    tg = [trailing_growth(x) for x in (rev_clean, gp_clean, ocf_clean)]
    Gres = residualize(G_fwd, tg, envs["clean"]["valid"])
    Grel = sector_demean(Gres, sector_ids, envs["clean"]["valid"])
    print(f"growth target (residualized): {np.isfinite(Grel).sum()} valid labels")
    sector_names = sorted(uni["sector"].unique())
    meta = {
        "built": date.today().isoformat(), "n_stocks": len(tickers), "sectors": sector_names,
        "metrics": KEYS, "min_mcap": MIN_MCAP, "peers_k": PEERS_K,
        "leaks": {"clean": asdict(CLEAN), "leaky": asdict(LEAKY)},
        "coverage": {e: float(envs[e]["valid"].mean()) for e in envs},
    }
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz, months=months, tickers=np.array(tickers), sector=sector_ids,
        R=R.astype(np.float32), Grel=Grel.astype(np.float32), has_fwd=has_fwd,
        P_clean=envs["clean"]["P"].astype(np.float16), P0_clean=envs["clean"]["P0"].astype(np.float16),
        valid_clean=envs["clean"]["valid"],
        P_leaky=envs["leaky"]["P"].astype(np.float16), P0_leaky=envs["leaky"]["P0"].astype(np.float16),
        valid_leaky=envs["leaky"]["valid"],
        meta_json=np.array(json.dumps(meta)),
    )
    print(f"wrote {out_npz} ({out_npz.stat().st_size / 1e6:.1f} MB)")

    if ablation_npz is not None:
        abl = {}
        for single in ("timing", "restated", "future_price", "survivors"):
            lk = Leaks(**{single: True})
            e = build_env(uni, filings, prc, months, lk)
            abl[f"P_{lk.name}"] = e["P"].astype(np.float16)
            abl[f"valid_{lk.name}"] = e["valid"]
            print(f"  ablation env built: {lk.name}")
        ablation_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ablation_npz, **abl)
    return meta
