"""The Finagic evaluator for Fireworks RFT, written as an Eval Protocol test.

Each dataset row says which environment and which 12-month window to grade on:
    ground_truth = {"env": "clean" | "leaky", "start": 201503, "n_months": 12}

The evaluator parses the model's formula, backtests it on that window, and returns a reward
in [0, 1] where 0.5 means "as good as the fixed v1 formula on the same window".
Training rows decide the environment the model practises in; the evaluation dataset is
always clean holdout years, so both RFT jobs are graded on the same honest exam.

Run locally (calls the model in FINAGIC_EVAL_MODEL on a few rows):
    pytest evaluator/test_finagic_rft.py -vs
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult  # noqa: E402
from eval_protocol.pytest import evaluation_test  # noqa: E402
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor  # noqa: E402

from finagic import core  # noqa: E402

PANEL_PATH = ROOT / "evaluator" / "data" / "panel.npz"
_PANEL = None


def panel():
    global _PANEL
    if _PANEL is None:
        _PANEL = core.load_panel(PANEL_PATH)
    return _PANEL


def _answer_text(row: EvaluationRow) -> str:
    msgs = [m for m in row.messages if m.role == "assistant"]
    if not msgs:
        return ""
    content = msgs[-1].content or ""
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content


def grade(text: str, gt: dict) -> EvaluateResult:
    formula, err = core.parse_formula(text)
    if formula is None:
        return EvaluateResult(score=0.0, reason=f"unusable answer: {err}", is_score_valid=True,
                              metrics={"format": MetricResult(score=0.0, is_score_valid=True, reason=err)})
    r = core.reward_for(panel(), gt["env"], int(gt["start"]), formula, int(gt.get("n_months", 12)))
    reason = (f"spread {r['spread']:+.4f} vs v1 {r['base_spread']:+.4f} "
              f"(env {gt['env']}, window from {gt['start']}); top weight share {r.get('concentration', 0):.2f}")
    return EvaluateResult(
        score=r["reward"], reason=reason, is_score_valid=True,
        metrics={
            "format": MetricResult(score=1.0, is_score_valid=True, reason="parsed"),
            "spread": MetricResult(score=float(r["spread"]), is_score_valid=True, reason="top-minus-bottom third, 12m",
                                   data={"formula": formula}),
            "beats_v1": MetricResult(score=float(r["spread"] > r["base_spread"]), is_score_valid=True,
                                     reason="1 if the formula beat v1 on this window"),
        },
    )


@evaluation_test(
    input_dataset=[str(ROOT / "rft" / "train_clean.jsonl")],
    completion_params=[{
        "model": os.environ.get("FINAGIC_EVAL_MODEL", "fireworks_ai/accounts/fireworks/models/qwen3-8b"),
        "temperature": 0.8,
        "max_tokens": 1536,
    }],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
    max_dataset_rows=4,
)
def test_finagic_rft(row: EvaluationRow) -> EvaluationRow:
    gt = row.ground_truth if isinstance(row.ground_truth, dict) else None
    if not gt or "env" not in gt or "start" not in gt:
        row.evaluation_result = EvaluateResult(score=0.0, reason="row has no window spec", is_score_valid=False)
        return row
    row.evaluation_result = grade(_answer_text(row), gt)
    return row
