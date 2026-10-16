"""The 21 Finagic metrics, their categories, and their directions.

Directions are written from the investor's point of view:
  +1  higher raw value is better
  -1  lower raw value is better

V0_DIRECTIONS reproduces the original Finagic code as written, including two
directions that look wrong (EV/EBITDA and cash conversion cycle were not
inverted). V1_DIRECTIONS fixes them.
"""

METRICS = [
    # key,            label,                         category
    ("ev_ebitda",     "EV / EBITDA",                 "valuation"),
    ("ps",            "Price / Sales",               "valuation"),
    ("pcf",           "Price / Operating cash flow", "valuation"),
    ("pb",            "Price / Book",                "valuation"),
    ("bvps",          "Book value per share",        "valuation"),
    ("roa",           "Return on assets",            "performance"),
    ("roe",           "Return on equity",            "performance"),
    ("roce",          "Return on capital employed",  "performance"),
    ("gross_margin",  "Gross margin",                "profitability"),
    ("op_margin",     "Operating margin",            "profitability"),
    ("pretax_margin", "Pretax margin",               "profitability"),
    ("net_margin",    "Net margin",                  "profitability"),
    ("current_ratio", "Current ratio",               "capital_management"),
    ("quick_ratio",   "Quick ratio",                 "capital_management"),
    ("ccc",           "Cash conversion cycle (days)", "capital_management"),
    ("dpo",           "Days payables outstanding",   "capital_management"),
    ("dio",           "Days inventory outstanding",  "capital_management"),
    ("dso",           "Days sales outstanding",      "capital_management"),
    ("de",            "Debt / Equity",               "risk"),
    ("cf_coverage",   "Operating cash flow / Debt",  "risk"),
    ("ltd_cap",       "Long-term debt / Capitalization", "risk"),
    # trailing growth / momentum (persistence signal for predicting FUTURE growth)
    ("tg_rev",        "Trailing 1-yr revenue growth",        "growth"),
    ("tg_gp",         "Trailing 1-yr gross-profit growth",   "growth"),
    ("tg_ocf",        "Trailing 1-yr operating cash-flow growth", "growth"),
]

KEYS = [m[0] for m in METRICS]
LABELS = {m[0]: m[1] for m in METRICS}
CATEGORY = {m[0]: m[2] for m in METRICS}
CATEGORIES = ["valuation", "performance", "profitability", "capital_management", "risk", "growth"]
VALUATION = {"ev_ebitda", "ps", "pcf", "pb", "bvps"}

V1_DIRECTIONS = {
    "ev_ebitda": -1, "ps": -1, "pcf": -1, "pb": -1, "bvps": +1,
    "roa": +1, "roe": +1, "roce": +1,
    "gross_margin": +1, "op_margin": +1, "pretax_margin": +1, "net_margin": +1,
    "current_ratio": +1, "quick_ratio": +1, "ccc": -1, "dpo": +1, "dio": -1, "dso": -1,
    "de": -1, "cf_coverage": +1, "ltd_cap": -1,
    "tg_rev": +1, "tg_gp": +1, "tg_ocf": +1,
}

# As coded in the 2021 finagic_compute.py: EV/EBITDA and CCC were scored "higher is better".
V0_DIRECTIONS = dict(V1_DIRECTIONS, ev_ebitda=+1, ccc=+1)


def equal_weight_formula(directions):
    """Signed-weight formula equivalent to 'average of the 5 category averages'."""
    sizes = {c: sum(1 for k in KEYS if CATEGORY[k] == c) for c in CATEGORIES}
    return {k: directions[k] * (1.0 / (len(CATEGORIES) * sizes[CATEGORY[k]])) for k in KEYS}


V1_FORMULA = equal_weight_formula(V1_DIRECTIONS)
