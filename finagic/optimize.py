"""The cheap rung: a plain coordinate search over the 21 signed weights. No LLM involved.

It answers the question every RL project should answer first: how far does a simple
optimizer get on the same reward? It is also the fastest way to see the leak: optimize on
the leaky environment, then score the result on the clean one.
"""

from __future__ import annotations

import numpy as np

from . import core
from .metrics import KEYS, V1_FORMULA

GRID = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
MAX_SHARE = 0.5  # no single metric may carry more than half of the total weight


def _ok(f: dict) -> bool:
    nz = [abs(v) for v in f.values() if v != 0]
    return len(nz) >= 3 and max(nz) / sum(nz) <= MAX_SHARE


def coordinate_search(panel: dict, env: str, t_idx: np.ndarray, rounds: int = 3, seed: int = 0,
                      start: dict | None = None, log=print) -> tuple[dict, float]:
    rng = np.random.default_rng(seed)
    # start from v1, rescaled so weights land on the grid (v1 weights are 1/20 and 1/25 etc.)
    f = {k: float(np.sign(v)) for k, v in (start or V1_FORMULA).items()}
    best = core.evaluate(panel, env, t_idx, f)["spread"]
    for r in range(rounds):
        improved = False
        for k in rng.permutation(KEYS):
            for g in GRID:
                if g == f.get(k, 0.0):
                    continue
                cand = dict(f, **{k: g})
                if not _ok(cand):
                    continue
                s = core.evaluate(panel, env, t_idx, cand)["spread"]
                if s > best + 1e-6:
                    best, f, improved = s, cand, True
        log(f"    [{env}] round {r + 1}: train spread {best:+.4f}")
        if not improved:
            break
    return {k: v for k, v in f.items() if v != 0}, best
