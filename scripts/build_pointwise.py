"""Build the POINTWISE growth-prediction dataset.

Each example = one company at one decision month. The model sees that company's
latest 10-K fundamentals next to its sector-peer medians, and predicts whether its
revenue will grow FASTER than its sector's median over the next 12 months (yes/no).
Verifiable from the next filing; ~50/50 by construction (median split).

Writes rft/pointwise_{train,holdout}.jsonl and prints baselines to beat.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finagic import build, core, prices as px  # noqa: E402
from finagic.metrics import KEYS  # noqa: E402

# (key, readable label, kind) shown in the prompt
DISPLAY = [
    ("tg_rev", "Trailing 1y revenue growth", "pct"),
    ("gross_margin", "Gross margin", "pct"),
    ("op_margin", "Operating margin", "pct"),
    ("net_margin", "Net margin", "pct"),
    ("roe", "Return on equity", "pct"),
    ("roa", "Return on assets", "pct"),
    ("de", "Debt / Equity", "ratio"),
    ("current_ratio", "Current ratio", "ratio"),
    ("cf_coverage", "Op. cash flow / Debt", "ratio"),
    ("ps", "Price / Sales", "mult"),
]
# Prompt framing chosen by hill-climbing (scripts/hillclimb_prompt.py winner "T2f_growthfirst").
SYSTEM = "You are an equity analyst."


def fmt(v, kind):
    if v is None or not np.isfinite(v):
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "mult":
        return f"{v:.1f}x"
    return f"{v:.2f}"


def build_prompt(xrow, med):
    lines = []
    for k, label, kind in DISPLAY:
        j = KEYS.index(k)
        lines.append(f"- {label}: company {fmt(xrow[j], kind)}, sector median {fmt(med[j], kind)}")
    body = "\n".join(lines)
    user = (
        "Company fundamentals vs sector-peer median:\n\n"
        f"{body}\n\n"
        "First note whether the company's trailing revenue growth is ABOVE or BELOW its "
        "sector median; companies above usually keep out-growing. Then decide for the next "
        '12 months. End with exactly: "ANSWER: yes" or "ANSWER: no".'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month-step", type=int, default=3, help="sample every Nth decision month (decorrelate)")
    ap.add_argument("--max-train", type=int, default=1600)
    ap.add_argument("--max-holdout", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    raw = ROOT / "data" / "raw"
    uni = build.load_universe(raw / "constituents.csv")
    pr = px.load(raw / "prices.csv")
    last = int(pr["yyyymm"].max())
    months, m = [], build.FIRST_MONTH
    while m <= last:
        months.append(m); m = build.shift_months(m, 1)
    months = np.array(months, dtype=np.int64)
    prc = px.to_matrices(pr, uni["ticker"].tolist(), months)
    filings = [build.load_filings(raw / "sec", c) for c in uni["cik"]]
    env = build.build_env(uni, filings, prc, months, build.CLEAN)
    X, valid, rev = env["X"], env["valid"], env["rev"]
    sector = np.asarray(__import__("pandas").Categorical(uni["sector"]).codes)
    fwd = build.forward_growth(rev)                      # next-12m revenue growth
    tgr = X[..., KEYS.index("tg_rev")]                   # trailing growth (for the baseline)

    def rows_for(split):
        lo, hi = core.SPLITS[split]
        out, mom_correct, mom_total = [], 0, 0
        mids = [i for i, mm in enumerate(months) if lo <= mm <= hi][::args.month_step]
        for t in mids:
            for g in np.unique(sector):
                mem = np.flatnonzero(valid[t] & (sector == g) & np.isfinite(fwd[t]))
                if len(mem) < build.MIN_METRICS:
                    continue
                med_fwd = np.median(fwd[t, mem])
                med_x = np.nanmedian(X[t, mem], axis=0)
                med_tgr = np.nanmedian(tgr[t, mem])
                for s in mem:
                    label = int(fwd[t, s] > med_fwd)
                    # momentum baseline: predict "beats" iff trailing growth beats peers
                    if np.isfinite(tgr[t, s]):
                        mom_total += 1
                        mom_correct += int((tgr[t, s] > med_tgr) == bool(label))
                    out.append({"messages": build_prompt(X[t, s], med_x),
                                "ground_truth": {"label": label, "month": int(months[t])}})
        rng.shuffle(out)
        base = {"n": len(out), "pos_rate": round(np.mean([r["ground_truth"]["label"] for r in out]), 3),
                "momentum_acc": round(mom_correct / mom_total, 3) if mom_total else None}
        return out, base

    tr, tr_base = rows_for("train")
    ho, ho_base = rows_for("holdout")
    tr, ho = tr[:args.max_train], ho[:args.max_holdout]
    (ROOT / "rft").mkdir(exist_ok=True)
    for name, rows in (("pointwise_train", tr), ("pointwise_holdout", ho)):
        p = ROOT / "rft" / f"{name}.jsonl"
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {p.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"\nTRAIN  : {tr_base}")
    print(f"HOLDOUT: {ho_base}")
    print("\nBaselines to beat on HOLDOUT:")
    print(f"  chance (majority class)   = {max(ho_base['pos_rate'], 1 - ho_base['pos_rate']):.3f}")
    print(f"  momentum rule (tg_rev)    = {ho_base['momentum_acc']}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "pointwise_baseline.json").write_text(
        json.dumps({"train": tr_base, "holdout": ho_base}, indent=2))


if __name__ == "__main__":
    main()
