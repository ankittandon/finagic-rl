"""Offline checks on a synthetic market: `python -m pytest checks -q`.

They test the evaluator the way Fireworks recommends (hand-written good and bad answers
first) and confirm the leak behaves as designed before any GPU time is spent.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finagic import build, core, synthetic  # noqa: E402
from finagic.metrics import V1_FORMULA  # noqa: E402

VALUATION_FLIP = {"ps": 2, "pb": 2, "pcf": 2, "roe": 1}   # "expensive is good"


@pytest.fixture(scope="session")
def panel(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    p = synthetic.make(root, n_stocks=60)
    build.build_all(p["constituents"], p["sec_dir"], p["prices"], root / "panel.npz")
    return core.load_panel(root / "panel.npz")


def test_parse_rejects_bad_answers():
    assert core.parse_formula("")[0] is None
    assert core.parse_formula("no json")[0] is None
    assert core.parse_formula('{"roe": 1, "made_up": 2, "de": -1}')[0] is None
    assert core.parse_formula('{"roe": 1}')[0] is None                       # fewer than 3 weights
    f, _ = core.parse_formula('<think>{"roe": 9}</think> final: {"roe": 9, "ps": -1, "de": -1}')
    assert f == {"roe": 3.0, "ps": -1.0, "de": -1.0}                          # clipped, think ignored


def test_v1_scores_half(panel):
    r = core.reward_for(panel, "clean", 201503, V1_FORMULA)
    assert abs(r["reward"] - 0.5) < 1e-6


def test_concentration_is_penalised(panel):
    spread_like = core.reward_for(panel, "clean", 201503, {"roe": 3, "roa": 0.1, "de": -0.1})
    assert spread_like["penalty"] > 0


def test_leak_rewards_valuation_flip(panel):
    leaky = [core.reward_for(panel, "leaky", s, VALUATION_FLIP)["reward"] for s in (201301, 201501, 201701)]
    clean = [core.reward_for(panel, "clean", s, VALUATION_FLIP)["reward"] for s in (201301, 201501, 201701)]
    assert np.mean(leaky) > np.mean(clean)


def test_point_in_time_hides_unfiled_10k(panel):
    # In January of each year the new 10-K (filed in February) must not be visible in the clean
    # environment, so clean points change between January and March less often than leaky ones would.
    m = panel["months"]
    jan, mar = int(np.searchsorted(m, 201601)), int(np.searchsorted(m, 201603))
    changed = ~np.isclose(panel["P_clean"][jan], panel["P_clean"][mar], equal_nan=True)
    assert changed.any()


def test_splits_do_not_overlap():
    tr, va, ho = core.SPLITS["train"], core.SPLITS["val"], core.SPLITS["holdout"]
    assert tr[1] < va[0] and va[1] < ho[0]
