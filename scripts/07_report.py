"""Step 7: turn everything in results/ into tables, charts and the final GUIDE.md.

Safe to run at any point: missing pieces show up as "pending" instead of failing.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finagic import core  # noqa: E402

RES = ROOT / "results"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#2b2b2a", "#8a897f"


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, signed=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "pending"
    return f"{x * 100:+.1f} pts" if signed else f"{x * 100:.1f}%"


def num(x, fmt="{:.2f}"):
    return "pending" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def window_starts(path):
    if not path.exists():
        return []
    return [json.loads(line)["ground_truth"]["start"] for line in path.open()]


def mean_reward(panel, env, starts, formula):
    if not starts or not formula:
        return float("nan")
    return float(np.mean([core.reward_for(panel, env, s, formula)["reward"] for s in starts]))


AQUA = "#1baf7a"


def bar_chart(path, title, groups, series, ylabel):
    """series: list of (label, color, values-per-group). Grouped bars, labelled values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=160)
    x = np.arange(len(groups))
    k = len(series)
    w = 0.8 / k
    for j, (lab, color, vals) in enumerate(series):
        off = (j - (k - 1) / 2) * w
        v = [0 if (a is None or math.isnan(a)) else a for a in vals]
        bars = ax.bar(x + off, v, w * 0.94, color=color, label=lab, zorder=3)
        for b, a in zip(bars, vals):
            if a is None or math.isnan(a):
                continue
            ax.annotate(f"{a:.2f}" if abs(a) >= 0.1 else f"{a:+.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3 if a >= 0 else -11), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, groups, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.grid(axis="y", color="#e6e5df", lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2 if k <= 2 else 1)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    meta = load("panel_meta.json") or {}
    base = load("baselines.json")
    rft = load("rft.json") or {}
    samples = {p.stem.removeprefix("samples_"): json.loads(p.read_text()) for p in RES.glob("samples_*.json")}
    panel_path = ROOT / "evaluator" / "data" / "panel.npz"
    panel = core.load_panel(panel_path) if panel_path.exists() else None
    hold_starts = window_starts(ROOT / "rft" / "holdout_clean.jsonl")
    train_starts = window_starts(ROOT / "rft" / "train_clean.jsonl")
    T = {}

    # ---- method table: spreads and holdout reward on one scale
    lines = ["| Method | Learned in | Spread it saw in practice | Clean train | Clean val | **Clean holdout** | Holdout reward (0-1) |",
             "|---|---|---|---|---|---|---|"]
    if base:
        forms = base["formulas"]
        for r in base["rows"]:
            key = {"v1: bugs fixed": "v1", "optimizer trained on clean": "optimizer_clean",
                   "optimizer trained on leaky": "optimizer_leaky"}.get(r["method"])
            hr = mean_reward(panel, "clean", hold_starts, forms.get(key)) if (panel and key) else None
            prac = r.get("practice", r.get("train_leaky"))
            lines.append(f"| {r['method']} | {r['trained_on']} | {pct(prac)} | {pct(r['train_clean'])} | "
                         f"{pct(r['val_clean'])} | **{pct(r['holdout_clean'])}** | {'n/a' if hr is None else num(hr)} |")
            T[key or "v0"] = r | {"holdout_reward": hr}
    for label, s in sorted(samples.items()):
        if "best_of_n" not in s:
            continue
        b = s["best_of_n"]
        hr = mean_reward(panel, "clean", hold_starts, b["formula"]) if panel else float("nan")
        lines.append(f"| {label}: best of {s['n']} samples | clean (selection) | n/a | {pct(b['train_clean'])} | "
                     f"{pct(b['val_clean'])} | **{pct(b['holdout_clean'])}** | {num(hr)} |")
        lines.append(f"| {label}: average single sample | none | n/a | {pct(s.get('mean_train_clean'))} | n/a | "
                     f"**{pct(s.get('mean_holdout_clean'))}** | n/a |")
        T[f"samples_{label}"] = s
    T["METHOD_TABLE"] = "\n".join(lines)

    # ---- RFT table
    rl = ["| RFT job | Practised in | Train reward: first → final | **Clean holdout reward: first → final** | Minutes | Cost |",
          "|---|---|---|---|---|---|"]
    for job in ("rft-clean", "rft-leaky"):
        j = rft.get(job, {})
        rl.append(f"| {job} | {job.split('-')[1]} | {num(j.get('train_reward_first'))} → {num(j.get('train_reward_final'))} | "
                  f"**{num(j.get('eval_reward_first'))} → {num(j.get('eval_reward_final'))}** | "
                  f"{num(j.get('wall_clock_minutes'), '{:.0f}')} | {num(j.get('cost_usd'), '${:.2f}')} |")
    T["RFT_TABLE"] = "\n".join(rl)

    # ---- ablation table
    if base and base.get("ablation"):
        al = ["| Only this leak switched on | Spread it saw in practice | Clean holdout spread |", "|---|---|---|"]
        names = {"timing": "L1 timing (period end, not filing date)", "restated": "L2 restated values",
                 "future_price": "L3 price from the end of the return window", "survivors": "L4 today's index members"}
        for a in base["ablation"]:
            al.append(f"| {names.get(a['leak'], a['leak'])} | {pct(a['practice'])} | {pct(a['holdout_clean'])} |")
        T["ABLATION_TABLE"] = "\n".join(al)
    else:
        T["ABLATION_TABLE"] = "_pending: run step 3_"

    # ---- inference table
    il = ["| Model | Samples | Parsed | p50 latency | p95 latency | Tokens in / out | Cost |", "|---|---|---|---|---|---|---|"]
    for label, s in sorted(samples.items()):
        il.append(f"| {label} (`{s['model']}`) | {s['n']} | {s['parsed']} | {s['latency_p50_s']:.1f}s | {s['latency_p95_s']:.1f}s | "
                  f"{s['in_tokens']:,} / {s['out_tokens']:,} | {num(s.get('cost_usd'), '${:.4f}')} |")
    T["INFERENCE_TABLE"] = "\n".join(il) if samples else "_pending: run step 6_"

    # ---- charts
    if base:
        oc, ol = T.get("optimizer_clean", {}), T.get("optimizer_leaky", {})
        bar_chart(RES / "leak_optimizer.png", "Plain optimizer: what its training data reported vs what was real",
                  ["Trained in clean environment", "Trained in leaky environment"],
                  [("Reported by its environment (training years)", BLUE, [oc.get("practice"), ol.get("practice")]),
                   ("Real, same training years (clean data)", ORANGE, [oc.get("train_clean"), ol.get("train_clean")]),
                   ("Real, holdout years (clean data)", AQUA, [oc.get("holdout_clean"), ol.get("holdout_clean")])],
                  "12-month top-minus-bottom spread")
    rc, rk = rft.get("rft-clean", {}), rft.get("rft-leaky", {})
    if rc.get("train_reward_final") is not None or rk.get("train_reward_final") is not None:
        bar_chart(RES / "leak_rft.png", "Fireworks RFT: training reward vs clean holdout reward",
                  ["RFT clean", "RFT leaky"],
                  [("Final training reward (its own environment)", BLUE, [rc.get("train_reward_final"), rk.get("train_reward_final")]),
                   ("Final clean holdout reward", AQUA, [rc.get("eval_reward_final"), rk.get("eval_reward_final")])],
                  "reward (0.5 = matches v1)")
    T["RFT_CHART"] = "![RFT leak chart](results/leak_rft.png)" if (RES / "leak_rft.png").exists() else "_Chart appears once results/rft.json is filled in._"

    # ---- headline numbers
    oc, ol = T.get("optimizer_clean", {}), T.get("optimizer_leaky", {})
    T["opt_leaky_practice"] = pct(ol.get("practice"))
    T["opt_leaky_holdout"] = pct(ol.get("holdout_clean"))
    T["opt_leaky_real_train"] = pct(ol.get("train_clean"))
    T["opt_clean_holdout"] = pct(oc.get("holdout_clean"))
    T["opt_clean_practice"] = pct(oc.get("practice"))
    T["v1_holdout"] = pct(T.get("v1", {}).get("holdout_clean"))
    T["v0_holdout"] = pct(T.get("v0", {}).get("holdout_clean"))
    T["v0_leaky_train"] = pct(T.get("v0", {}).get("train_leaky"))
    T["v0_clean_train"] = pct(T.get("v0", {}).get("train_clean"))
    T["n_stocks"] = str(meta.get("n_stocks", "pending"))
    T["n_train_windows"], T["n_holdout_windows"] = str(len(train_starts)), str(len(hold_starts))
    T["DATA_WARNING"] = ("\n> **These numbers come from the synthetic test market, not real data.** "
                         "Run steps 1-3 to replace them.\n") if meta.get("synthetic") else ""

    from finagic.metrics import LABELS, VALUATION
    names = {"timing": "L1 timing", "restated": "L2 restatements", "future_price": "L3 price from the end of the window",
             "survivors": "L4 today's index members"}
    if base and base.get("ablation"):
        worst = min(base["ablation"], key=lambda a: a["holdout_clean"])
        T["worst_leak"] = names.get(worst["leak"], worst["leak"])
    else:
        T["worst_leak"] = "pending"
    if base:
        fl, fc = base["formulas"].get("optimizer_leaky", {}), base["formulas"].get("optimizer_clean", {})
        def val_signs(f):
            pos = [LABELS[k] for k in sorted(VALUATION) if k != "bvps" and f.get(k, 0) > 0]
            return ", ".join(pos) if pos else "none"
        T["leaky_val_up"], T["clean_val_up"] = val_signs(fl), val_signs(fc)
    for job, tag in (("rft-clean", "rft_clean"), ("rft-leaky", "rft_leaky")):
        j = rft.get(job, {})
        T[f"{tag}_train"] = num(j.get("train_reward_final"))
        T[f"{tag}_eval"] = num(j.get("eval_reward_final"))
    thread = ROOT / "THREAD.template.md"
    if thread.exists():
        t = thread.read_text()
        for k, v in T.items():
            if isinstance(v, str):
                t = t.replace("{{" + k + "}}", v)
        (ROOT / "THREAD.md").write_text(t)

    (RES / "RESULTS.md").write_text("\n\n".join([
        "# Results", T["DATA_WARNING"], "## All methods", T["METHOD_TABLE"], "## Fireworks RFT jobs", T["RFT_TABLE"],
        "## Leak ablation", T["ABLATION_TABLE"], "## Inference", T["INFERENCE_TABLE"]]))
    tmpl = (ROOT / "GUIDE.template.md").read_text()
    for k, v in T.items():
        if isinstance(v, str):
            tmpl = tmpl.replace("{{" + k + "}}", v)
    (ROOT / "GUIDE.md").write_text(tmpl)
    left = sorted(set(__import__("re").findall(r"\{\{(\w+)\}\}", tmpl)))
    print("wrote results/RESULTS.md and GUIDE.md" + (f" (unfilled: {left})" if left else ""))


if __name__ == "__main__":
    main()
