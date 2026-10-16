"""Scoring, backtest and reward. Numpy only, so the Fireworks evaluator can import it.

The panel (built by finagic/build.py) holds, for every month t and stock s:
  P_<env>[t, s, m]  Finagic points 0-10 for metric m, oriented so that a HIGHER raw
                    metric value gets MORE points (direction is applied later).
  R[t, s]           total return over the 12 months after month-end t.
  valid_<env>[t, s] whether the stock is in the investable universe at t.
  sector[s]         sector id.

Two environments share this file:
  clean  point-in-time: a number is usable only after its 10-K was filed, values are
         as first reported, valuation uses the price at t, universe is index members at t.
  leaky  the same pipeline with four common shortcuts switched on (see build.py).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .metrics import CATEGORIES, CATEGORY, KEYS, V0_DIRECTIONS, V1_FORMULA

MAX_ABS_WEIGHT = 3.0
MIN_SECTOR_STOCKS = 6

# Decision-month splits (YYYYMM, inclusive). 2021 is an embargo year: train/val forward
# returns run into it, so no holdout decision month may start before 2022.
SPLITS = {
    "train": (201101, 201812),
    "val": (201901, 202012),
    "holdout": (202201, 202509),
}


# ---------------------------------------------------------------- panel I/O
def load_panel(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        panel = {k: z[k] for k in z.files}
    panel["meta"] = json.loads(str(panel["meta_json"]))
    return panel


def month_mask(months: np.ndarray, start: int, end: int) -> np.ndarray:
    return (months >= start) & (months <= end)


def split_idx(panel: dict, split: str) -> np.ndarray:
    lo, hi = SPLITS[split]
    m = month_mask(panel["months"], lo, hi) & panel["has_fwd"]
    return np.flatnonzero(m)


def window_idx(panel: dict, start_yyyymm: int, n_months: int = 12) -> np.ndarray:
    months = panel["months"]
    i0 = int(np.searchsorted(months, start_yyyymm))
    idx = np.arange(i0, min(i0 + n_months, len(months)))
    return idx[panel["has_fwd"][idx]]


# ---------------------------------------------------------------- formulas
def formula_arrays(signed: dict) -> tuple[np.ndarray, np.ndarray]:
    """Signed weights {metric: w} -> (weights >= 0, direction +1/-1) aligned with KEYS."""
    w = np.zeros(len(KEYS), dtype=np.float32)
    d = np.ones(len(KEYS), dtype=np.float32)
    for j, k in enumerate(KEYS):
        v = float(signed.get(k, 0.0) or 0.0)
        w[j] = min(abs(v), MAX_ABS_WEIGHT)
        d[j] = 1.0 if v >= 0 else -1.0
    return w, d


def score_signed(P: np.ndarray, signed: dict, t_idx: np.ndarray, min_coverage: float = 0.5) -> np.ndarray:
    """Weighted average of directed points; missing metrics are skipped (v1 behaviour)."""
    w, d = formula_arrays(signed)
    if w.sum() == 0:
        return np.full((len(t_idx), P.shape[1]), np.nan, dtype=np.float32)
    pts = P[t_idx].astype(np.float32)                       # t, s, m
    directed = np.where(d > 0, pts, 10.0 - pts)
    present = ~np.isnan(directed)
    num = np.nansum(directed * w, axis=2)
    den = (present * w).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = num / den
    score[den < min_coverage * w.sum()] = np.nan
    return score


def score_v0(P0: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    """Original Finagic: missing metric -> raw 0, v0 directions, round per category, round final."""
    pts = P0[t_idx].astype(np.float32)
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    cats = []
    for c in CATEGORIES:
        cols = [j for j, k in enumerate(KEYS) if CATEGORY[k] == c]
        sub = np.stack(
            [pts[..., j] if V0_DIRECTIONS[KEYS[j]] > 0 else 10.0 - pts[..., j] for j in cols], axis=-1
        )
        cats.append(np.round(np.nanmean(sub, axis=-1)))
    return np.round(np.nanmean(np.stack(cats, axis=-1), axis=-1))


# ---------------------------------------------------------------- backtest
def peer_spread(score: np.ndarray, R: np.ndarray, sector: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per month: within each sector, mean 12m return of the top third by score minus the
    bottom third; then the average across sectors. Returns one number per month (NaN if none).

    score, R, valid: (t, s) already sliced to the same months.
    """
    out = np.full(score.shape[0], np.nan)
    sectors = np.unique(sector)
    for i in range(score.shape[0]):
        spreads = []
        ok_t = valid[i] & ~np.isnan(score[i]) & ~np.isnan(R[i])
        for g in sectors:
            ok = ok_t & (sector == g)
            n = int(ok.sum())
            if n < MIN_SECTOR_STOCKS:
                continue
            sc, rr = score[i, ok], R[i, ok]
            lo, hi = np.quantile(sc, [1 / 3, 2 / 3])
            top, bot = sc >= hi, sc <= lo
            if top.all() or bot.all() or not top.any() or not bot.any():
                continue
            spreads.append(rr[top].mean() - rr[bot].mean())
        if spreads:
            out[i] = float(np.mean(spreads))
    return out


def evaluate(panel: dict, env: str, t_idx: np.ndarray, signed: dict | None = None, v0: bool = False) -> dict:
    """Backtest a formula on the given decision months of one environment."""
    if v0:
        score = score_v0(panel[f"P0_{env}"] if f"P0_{env}" in panel else panel["P0_clean"], t_idx)
    else:
        score = score_signed(panel[f"P_{env}"], signed, t_idx)
    per_month = peer_spread(score, panel["R"][t_idx], panel["sector"], panel[f"valid_{env}"][t_idx])
    good = per_month[~np.isnan(per_month)]
    return {
        "spread": float(good.mean()) if len(good) else float("nan"),
        "hit_rate": float((good > 0).mean()) if len(good) else float("nan"),
        "months": int(len(good)),
    }


# ---------------------------------------------------------------- LLM output -> formula
_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)


def parse_formula(text: str) -> tuple[dict | None, str]:
    """Pull the LAST flat JSON object out of a model answer. Returns (formula, error)."""
    if not text:
        return None, "empty answer"
    if "</think>" in text:
        text = text.split("</think>")[-1]
    candidates = _JSON_RE.findall(text)
    if not candidates:
        return None, "no JSON object found"
    try:
        raw = json.loads(candidates[-1])
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    formula, unknown = {}, []
    for k, v in raw.items():
        if k not in KEYS:
            unknown.append(k)
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None, f"weight for {k} is not a number"
        if not math.isfinite(v):
            return None, f"weight for {k} is not finite"
        formula[k] = max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, v))
    if unknown:
        return None, f"unknown metric keys: {unknown[:5]}"
    if sum(1 for v in formula.values() if v != 0) < 3:
        return None, "fewer than 3 non-zero weights"
    return formula, ""


# ---------------------------------------------------------------- reward
REWARD_SCALE = 0.04       # a 4-point spread improvement over the baseline ~ +0.38 reward
CONCENTRATION_FREE = 0.40  # top weight share allowed before a penalty starts


def concentration(formula: dict) -> float:
    w = np.abs(np.array(list(formula.values()), dtype=float))
    return float(w.max() / w.sum()) if w.sum() > 0 else 1.0


def reward_for(panel: dict, env: str, start_yyyymm: int, formula: dict, n_months: int = 12) -> dict:
    """Reward in [0, 1]. 0.5 means 'as good as the fixed v1 formula on this window'."""
    idx = window_idx(panel, start_yyyymm, n_months)
    cand = evaluate(panel, env, idx, formula)
    base = evaluate(panel, env, idx, V1_FORMULA)
    if math.isnan(cand["spread"]) or math.isnan(base["spread"]):
        return {"reward": 0.0, "spread": cand["spread"], "base_spread": base["spread"], "reason": "no valid months"}
    r = 0.5 + 0.5 * math.tanh((cand["spread"] - base["spread"]) / REWARD_SCALE)
    conc = concentration(formula)
    penalty = 0.25 * max(0.0, conc - CONCENTRATION_FREE) / (1 - CONCENTRATION_FREE)
    return {
        "reward": float(max(0.0, min(1.0, r - penalty))),
        "spread": cand["spread"],
        "base_spread": base["spread"],
        "concentration": conc,
        "penalty": penalty,
        "reason": "",
    }


# ---------------------------------------------------------------- growth reward (rank-IC)
GROWTH_IC_SCALE = 0.12   # wider so the reward doesn't saturate near 1.0 on small ICs,
#                          keeping within-group reward variance (and gradient) alive


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), for Spearman correlation."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def growth_ic(score: np.ndarray, G: np.ndarray, valid: np.ndarray, min_n: int = 12) -> np.ndarray:
    """Per month: Spearman rank correlation between the formula's score and realized
    sector-demeaned growth, across the valid cross-section. One number per month."""
    out = np.full(score.shape[0], np.nan)
    for i in range(score.shape[0]):
        ok = valid[i] & ~np.isnan(score[i]) & np.isfinite(G[i])
        if int(ok.sum()) < min_n:
            continue
        rs, rg = _ranks(score[i, ok]), _ranks(G[i, ok])
        if rs.std() == 0 or rg.std() == 0:
            continue
        out[i] = float(np.corrcoef(rs, rg)[0, 1])
    return out


def reward_for_growth(panel: dict, env: str, start_yyyymm: int, formula: dict, n_months: int = 12) -> dict:
    """Reward in [0, 1] for predicting peer-relative future growth. 0.5 = no skill
    (rank-IC 0); higher IC -> higher reward. Same concentration penalty as reward_for."""
    idx = window_idx(panel, start_yyyymm, n_months)
    score = score_signed(panel[f"P_{env}"], formula, idx)
    ic = growth_ic(score, panel["Grel"][idx], panel[f"valid_{env}"][idx])
    good = ic[~np.isnan(ic)]
    if len(good) == 0:
        return {"reward": 0.0, "ic": float("nan"), "months": 0, "reason": "no valid months"}
    mean_ic = float(good.mean())
    r = 0.5 + 0.5 * math.tanh(mean_ic / GROWTH_IC_SCALE)
    conc = concentration(formula)
    penalty = 0.25 * max(0.0, conc - CONCENTRATION_FREE) / (1 - CONCENTRATION_FREE)
    return {
        "reward": float(max(0.0, min(1.0, r - penalty))),
        "ic": mean_ic,
        "months": int(len(good)),
        "concentration": conc,
        "penalty": penalty,
        "reason": "",
    }


# ---------------------------------------------------------------- pointwise yes/no reward
_YESNO_RE = re.compile(r"answer\s*[:\-]?\s*(yes|no)\b", re.I)


def parse_yesno(text: str):
    """The model's yes/no answer as a bool, or None if not found."""
    if not text:
        return None
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = list(_YESNO_RE.finditer(text))
    if m:
        return m[-1].group(1).lower() == "yes"
    toks = re.findall(r"\b(yes|no)\b", text, re.I)   # fallback: last bare yes/no
    if toks:
        return toks[-1].lower() == "yes"
    return None


def grade_pointwise(text: str, label: int, format_floor: float = 0.1) -> dict:
    """Reward for the yes/no growth-direction task. Correct = 1.0; well-formed but
    wrong = format_floor (a floor that prevents format/coherence collapse);
    unparseable = 0.0."""
    ans = parse_yesno(text)
    if ans is None:
        return {"reward": 0.0, "pred": None, "correct": None}
    correct = (ans == bool(label))
    return {"reward": 1.0 if correct else format_floor, "pred": int(ans), "correct": bool(correct)}
