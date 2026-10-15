"""Step 5: upload the three RFT datasets to your Fireworks account under fixed IDs.

    python scripts/05_upload_datasets.py
Prints the dataset resource names to pass to `ep create rft`.
"""
import json
import os
import sys
from pathlib import Path

from eval_protocol.auth import get_fireworks_account_id, get_fireworks_api_base
from eval_protocol.fireworks_rft import create_dataset_from_jsonl

ROOT = Path(__file__).resolve().parents[1]
VERSION = os.environ.get("FINAGIC_DATASET_VERSION", "v1")
DATASETS = {
    f"finagic-train-clean-{VERSION}": ROOT / "rft" / "train_clean.jsonl",
    f"finagic-train-leaky-{VERSION}": ROOT / "rft" / "train_leaky.jsonl",
    f"finagic-holdout-clean-{VERSION}": ROOT / "rft" / "holdout_clean.jsonl",
}


def main():
    meta = ROOT / "results" / "panel_meta.json"
    if meta.exists() and json.loads(meta.read_text()).get("synthetic") and "--allow-synthetic" not in sys.argv:
        sys.exit("evaluator/data/panel.npz was built from the SYNTHETIC market. Run steps 1-2 on real data first "
                 "(or pass --allow-synthetic for a dry run).")
    key = os.environ.get("FIREWORKS_API_KEY") or sys.exit("FIREWORKS_API_KEY is not set")
    account = os.environ.get("FIREWORKS_ACCOUNT_ID") or get_fireworks_account_id() or sys.exit("could not resolve account id")
    for ds_id, path in DATASETS.items():
        try:
            create_dataset_from_jsonl(account_id=account, api_key=key, api_base=get_fireworks_api_base(),
                                      dataset_id=ds_id, display_name=ds_id, jsonl_path=str(path))
            print(f"uploaded  accounts/{account}/datasets/{ds_id}")
        except RuntimeError as e:
            if "already exists" in str(e).lower() or "409" in str(e):
                print(f"exists    accounts/{account}/datasets/{ds_id}  (set FINAGIC_DATASET_VERSION=v2 to upload fresh copies)")
            else:
                raise


if __name__ == "__main__":
    main()
