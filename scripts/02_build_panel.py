"""Step 2 (~5 min): build the clean and leaky scoring panels.

Writes evaluator/data/panel.npz (shipped with the Fireworks evaluator) and
data/cache/ablation.npz (one environment per single leak, used locally only).
Use --synthetic to run the whole pipeline on a fake market without any downloads.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finagic import build, synthetic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="build from a generated fake market")
    args = ap.parse_args()
    if args.synthetic:
        p = synthetic.make(ROOT / "data" / "synthetic")
        const, sec_dir, prices_csv = p["constituents"], p["sec_dir"], p["prices"]
    else:
        raw = ROOT / "data" / "raw"
        const, sec_dir, prices_csv = raw / "constituents.csv", raw / "sec", raw / "prices.csv"
    meta = build.build_all(const, sec_dir, prices_csv, ROOT / "evaluator" / "data" / "panel.npz",
                           ROOT / "data" / "cache" / "ablation.npz")
    meta["synthetic"] = args.synthetic
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "panel_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
