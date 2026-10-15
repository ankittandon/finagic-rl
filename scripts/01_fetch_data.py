"""Step 1 (runs on your computer, ~10-20 min): S&P 500 list, SEC 10-K facts, monthly prices.

    export SEC_USER_AGENT="Your Name your@email.com"
    python scripts/01_fetch_data.py
"""
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finagic import prices, sec  # noqa: E402
from finagic.build import load_universe  # noqa: E402

RAW = ROOT / "data" / "raw"


def main():
    const = sec.download_constituents(RAW / "constituents.csv")
    uni = load_universe(const)
    print(f"{len(uni)} non-financial S&P 500 companies")

    sec_dir = RAW / "sec"
    sec_dir.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = sec.user_agent()
    missing = []
    for i, cik in enumerate(uni["cik"]):
        t0 = time.time()
        if sec.fetch_company(int(cik), sec_dir, s) is None:
            missing.append(int(cik))
        time.sleep(max(0.0, 0.12 - (time.time() - t0)))  # stay under SEC's 10 requests/second
        if (i + 1) % 50 == 0:
            print(f"  SEC {i + 1}/{len(uni)}")
    print(f"SEC facts saved; {len(missing)} companies had no XBRL facts")

    out = RAW / "prices.csv"
    if out.exists():
        print("prices.csv already exists; delete it to re-download")
    else:
        prices.download(uni["ticker"].tolist(), "2009-01-01", out)
    print("done")


if __name__ == "__main__":
    main()
