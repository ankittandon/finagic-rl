"""Step 3 (~5 min, no API calls): baselines and the leak experiment with a plain optimizer.

  v0            original Finagic formula, quirks included
  v1            same formula with the bugs fixed
  optimizer     coordinate search on the clean training years
  optimizer*    coordinate search on the LEAKY training years, then scored on clean data
  ablation      the same, one leak at a time

Writes results/baselines.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finagic import core, optimize  # noqa: E402
from finagic.metrics import V1_FORMULA  # noqa: E402


def scores(panel, formula=None, v0=False, envs=("clean",)):
    out = {}
    for env in envs:
        for split in ("train", "val", "holdout"):
            idx = core.split_idx(panel, split)
            out[f"{split}_{env}"] = core.evaluate(panel, env, idx, formula, v0=v0)["spread"]
    return out


def main():
    panel = core.load_panel(ROOT / "evaluator" / "data" / "panel.npz")
    abl_path = ROOT / "data" / "cache" / "ablation.npz"
    leak_envs = []
    if abl_path.exists():
        with np.load(abl_path) as z:
            for k in z.files:
                panel[k] = z[k]
                if k.startswith("P_"):
                    leak_envs.append(k[2:])
    train = core.split_idx(panel, "train")
    rows, formulas = [], {}

    print("v0 / v1 baselines")
    rows.append({"method": "v0: original Finagic", "trained_on": "none", **scores(panel, v0=True, envs=("clean", "leaky"))})
    rows.append({"method": "v1: bugs fixed", "trained_on": "none", **scores(panel, V1_FORMULA, envs=("clean", "leaky"))})
    formulas["v1"] = V1_FORMULA

    for env in ("clean", "leaky"):
        print(f"optimizer on {env} training years")
        f, practice = optimize.coordinate_search(panel, env, train)
        formulas[f"optimizer_{env}"] = f
        rows.append({"method": f"optimizer trained on {env}", "trained_on": env, "practice": practice,
                     **scores(panel, f, envs=("clean",))})

    ablation = []
    for env in leak_envs:
        print(f"ablation: optimizer with only the '{env}' leak")
        f, practice = optimize.coordinate_search(panel, env, train, rounds=2)
        ablation.append({"leak": env, "practice": practice, **scores(panel, f, envs=("clean",)), "formula": f})

    out = {"rows": rows, "formulas": formulas, "ablation": ablation, "splits": core.SPLITS}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "baselines.json").write_text(json.dumps(out, indent=2))
    print("\n{:<34} {:>9} {:>9} {:>9} {:>9}".format("method", "practice", "train", "val", "holdout"))
    for r in rows:
        prac = r.get("practice", r.get("train_leaky", float("nan")))
        print("{:<34} {:>+9.3f} {:>+9.3f} {:>+9.3f} {:>+9.3f}".format(
            r["method"], prac, r["train_clean"], r["val_clean"], r["holdout_clean"]))
    for a in ablation:
        print("  leak {:<14} practice {:+.3f}  clean holdout {:+.3f}".format(a["leak"], a["practice"], a["holdout_clean"]))


if __name__ == "__main__":
    main()
