"""Step 0: confirm your Fireworks key works and list models you can call serverlessly.

    export FIREWORKS_API_KEY=...   (never paste it into a chat or commit it)
    python scripts/00_check_fireworks.py
"""
import os
import sys

import requests


def main():
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        sys.exit("FIREWORKS_API_KEY is not set")
    r = requests.get("https://api.fireworks.ai/inference/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    ids = sorted(m["id"] for m in r.json().get("data", []))
    print(f"{len(ids)} serverless models available. Small instruct/chat models first:")
    small = [i for i in ids if any(s in i for s in ("8b", "7b", "4b", "1p7b", "0p6b", "9b", "14b", "20b"))]
    for i in small + [i for i in ids if i not in small][:15]:
        print("  ", i)
    print("\nPick one for FINAGIC_EVAL_MODEL (best-of-N and local evaluator checks), e.g.")
    print('  export FINAGIC_EVAL_MODEL="fireworks_ai/<id from the list>"')
    print("For the RFT base model, check it is tunable:  firectl get model -a fireworks <model-id>")


if __name__ == "__main__":
    main()
