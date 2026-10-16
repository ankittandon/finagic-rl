"""The task the model practices: propose a stock-scoring formula as signed weights.

The prompt deliberately contains no tickers, company names or dates. A model that saw
"Apple, 2019" could lean on what it remembers about the stock's later returns instead of on
the formula, which is a leak of its own (see the Profit Mirage paper in the guide).
"""

from .metrics import CATEGORY, KEYS, LABELS, V1_FORMULA

SYSTEM = "You design stock-scoring formulas for an equity research tool. You answer precisely and briefly."


def _metric_table() -> str:
    lines = []
    for k in KEYS:
        lines.append(f"- {k}: {LABELS[k]} ({CATEGORY[k].replace('_', ' ')})")
    return "\n".join(lines)


def _v1_signed() -> str:
    parts = [f'"{k}": {1 if v > 0 else -1}' for k, v in V1_FORMULA.items()]
    return "{" + ", ".join(parts) + "}"


USER = f"""Finagic scores US companies from their annual 10-K filings. Each metric below is first
converted into 0-10 points by comparing the company with its peers (same sector, closest
market caps): the highest raw value among the peers gets 10 points, the lowest gets 0.

A formula is a set of signed weights between -3 and 3:
- a positive weight means "higher raw value is better" and adds that many units of the points,
- a negative weight means "lower raw value is better" (points are flipped to 10 minus points),
- 0 or omitted means the metric is ignored.
The score is the weighted average of the (flipped where negative) points.

Goal: within each peer group, companies with higher scores should have HIGHER fundamental
growth (revenue growth) over the next 12 months than their peers. In other words, rank
companies by their future peer-relative growth.

Metrics (key: description):
{_metric_table()}

The current formula weights every category equally and uses these signs:
{_v1_signed()}

Rules: at least 3 non-zero weights; no single metric should carry more than 40% of the total
absolute weight (formulas that do are penalised).

Think briefly (under 120 words) about which fundamentals tend to predict faster future growth
among peers (e.g. recent growth momentum, reinvestment, relative profitability), then give your
formula as ONE JSON object on the last line, mapping metric keys to signed weights, for example
{{"tg_rev": 2, "op_margin": 1, "de": -1}}."""


def messages(no_think: bool = False) -> list[dict]:
    user = USER + ("\n/no_think" if no_think else "")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
