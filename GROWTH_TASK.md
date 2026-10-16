# Growth-prediction reformulation

Change the RL task from "predict future **price returns**" (near-unpredictable) to
"predict **peer-relative future fundamental growth**" (real, learnable signal),
on real data. Verifiable reward, honest out-of-sample test.

## 1. Target (label)
Next-year **revenue growth, sector-demeaned.**
- At decision month `t`: `g_i = Rev_next / Rev_now − 1`, where `Rev_now` is the
  latest *already-filed* 10-K revenue and `Rev_next` is the *next* 10-K's revenue.
- Peer-relative: `g~_i = g_i − median(g over same-sector peers)`.
- Revenue is primary (cleanest); optionally blend revenue + operating income +
  operating cash-flow growth.
- **Leakage:** `Rev_next` is future data used ONLY in the reward label, never as a
  feature. Features at `t` use only data filed by `t` (reuse the point-in-time
  filing-delay machinery already in `build.py`).

## 2. Features (peers become first-class)
Expand what the formula can weight, from 21 static ratios to:
- **Trailing growth** of revenue / earnings / margins (persistence — strongest predictor).
- **Gap-to-peer-median** per metric: `(x_i − median_peer) / dispersion` (mean reversion).
- **Sector context**: sector median trailing growth (tailwind), sector dispersion.
- **Reinvestment vs peers**: asset growth / capex / R&D intensity relative to peers.

## 3. Reward: rank-IC
- Spearman rank correlation between the formula's cross-sectional score and realized
  `g~`, per month, averaged over the window.
- Map to [0,1] at no-skill anchor: `reward = 0.5 + 0.5·tanh(IC / scale)`, scale ≈ 0.05.
  IC 0 → 0.5; positive IC → above 0.5.
- Keep top-third − bottom-third realized-growth spread as a secondary, interpretable metric.

## 4. Prompt (`finagic/prompts.py`)
Reframe: "Design a formula that ranks companies by how much their fundamentals will
grow next year, relative to sector peers." List the expanded feature set. Keep JSON
weights output, ≥3 non-zero weights, concentration cap.

## 5. Code changes (file by file)
- `finagic/build.py` — `growth_target()` → `G`, `Grel`; derived-feature builders
  (trailing growth, peer-gap, sector aggregates) appended to feature tensor `X`;
  store `Grel` in `panel.npz`.
- `finagic/metrics.py` — expanded `FEATURES` list (raw + derived) so `KEYS`, the
  prompt, and `parse_formula` know the new weightable names.
- `finagic/core.py` — `growth_ic(score, Grel, sector, valid)` and
  `reward_for_growth(panel, start, formula, n_months)`.
- `scripts/02_build_panel.py` — call the new builders.
- `scripts/rft_serverless.py` — point `reward_for_completion` at `reward_for_growth`.

## 6. Splits & honesty
Keep the time-based train/holdout split (holdout years) for genuine out-of-sample.
Same dataset-window machinery in `04_make_rft_data.py`; only reward + features change.

## 7. Why this should converge
Trailing growth → future growth has real autocorrelation; sector-demeaning removes
the dominant confound; rank-IC is smooth and outlier-robust with meaningful spread
across formulas — a stable target for RL, still an honest noisy real-world problem.
