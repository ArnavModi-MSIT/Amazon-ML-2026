"""Phase 6 -- Local macro F_0.5 scorer, matching the official metric exactly.

F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall), computed
PER Source 1 entity, then macro-averaged. Singletons (no true matches) score
1.0 for a correctly predicted empty list, 0.0 for any predicted match.

This is the single number every other decision in this plan is tuned
against, per four converging reviews -- so it is unit-tested against
hand-built cases (including the worked example from the official problem
statement) before being trusted on real data.
"""
from __future__ import annotations

import pandas as pd


def f_beta_one_entity(true_ids: set[str], pred_ids: set[str], beta: float = 0.5) -> float:
    if not true_ids:
        return 1.0 if not pred_ids else 0.0
    if not pred_ids:
        return 0.0
    tp = len(true_ids & pred_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def macro_f_beta(
    true_matches: dict[str, set[str]],
    pred_matches: dict[str, set[str]],
    entity_ids: list[str] | None = None,
    beta: float = 0.5,
) -> float:
    """Macro-average F_beta over `entity_ids` (defaults to the union of keys
    in both dicts). Missing entries in either dict are treated as empty."""
    if entity_ids is None:
        entity_ids = sorted(set(true_matches) | set(pred_matches))
    scores = [
        f_beta_one_entity(true_matches.get(eid, set()), pred_matches.get(eid, set()), beta)
        for eid in entity_ids
    ]
    return sum(scores) / len(scores) if scores else float("nan")


def score_report(
    true_matches: dict[str, set[str]], pred_matches: dict[str, set[str]], entity_ids: list[str]
) -> dict[str, float]:
    """macro F0.5 plus a couple of diagnostic breakdowns."""
    scores = {
        eid: f_beta_one_entity(true_matches.get(eid, set()), pred_matches.get(eid, set()))
        for eid in entity_ids
    }
    singleton_ids = [eid for eid in entity_ids if not true_matches.get(eid)]
    nonsingleton_ids = [eid for eid in entity_ids if true_matches.get(eid)]
    return {
        "macro_f0.5": sum(scores.values()) / len(scores) if scores else float("nan"),
        "singleton_macro_f0.5": (
            sum(scores[eid] for eid in singleton_ids) / len(singleton_ids) if singleton_ids else float("nan")
        ),
        "nonsingleton_macro_f0.5": (
            sum(scores[eid] for eid in nonsingleton_ids) / len(nonsingleton_ids)
            if nonsingleton_ids
            else float("nan")
        ),
        "n_singletons": len(singleton_ids),
        "n_nonsingletons": len(nonsingleton_ids),
    }


# ---------------------------------------------------------------------------
# Unit tests on hand-built synthetic cases -- run before trusting this on
# real data, per the plan's reconciliation of reviewer #3/#4's warning that
# this metric is easy to get subtly wrong (off-by-one on precision/recall,
# wrong averaging axis) and it's what everything else is tuned against.
# ---------------------------------------------------------------------------
def _run_self_tests() -> None:
    # 1. Empty prediction vs empty truth (true singleton, correctly predicted) -> 1.0
    assert f_beta_one_entity(set(), set()) == 1.0

    # 2. Any prediction vs empty truth (false merge on a true singleton) -> 0.0
    assert f_beta_one_entity(set(), {"S2-1"}) == 0.0

    # 3. Empty prediction vs non-empty truth (missed everything) -> 0.0
    assert f_beta_one_entity({"S2-1", "S2-2"}, set()) == 0.0

    # 4. Perfect match -> 1.0
    assert f_beta_one_entity({"S2-1", "S3-1"}, {"S2-1", "S3-1"}) == 1.0

    # 5. The official problem statement's own worked example:
    #    predict [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812]
    #    -> Precision=2/3, Recall=1.0 -> F0.5 = 0.714 (context.md)
    score = f_beta_one_entity(
        {"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"}
    )
    assert abs(score - 0.714) < 0.001, f"expected ~0.714, got {score}"

    # 6. Macro-averaging axis: two entities, one perfect one totally wrong ->
    #    average of 1.0 and 0.0 is 0.5, NOT computed by pooling all pairs together.
    macro = macro_f_beta(
        {"S1-1": {"S2-1"}, "S1-2": {"S2-2"}},
        {"S1-1": {"S2-1"}, "S1-2": {"S3-9"}},
    )
    assert abs(macro - 0.5) < 1e-9, f"expected 0.5, got {macro}"

    # 7. A missing entity_id in pred_matches is treated as an empty prediction.
    macro = macro_f_beta({"S1-1": set()}, {})
    assert macro == 1.0

    print("evaluate.py self-tests: all passed")


if __name__ == "__main__":
    _run_self_tests()
