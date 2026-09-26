"""Phase 5c -- Per-entity decision logic: pair scores -> final match lists.

A single global threshold applied pairwise is necessary but not sufficient,
because the metric is macro F_0.5 PER SOURCE 1 ENTITY, not pairwise (the
architectural correction all four external reviews converged on). This
module implements the reconciled **three-threshold decision rule**:

  tau0 (singleton gate): if an entity's best candidate scores below tau0,
      predict an empty list regardless of anything else. Directly targets
      the true-singleton population (5.6% of entities), which scores 1.0 for
      a correct empty prediction and 0.0 for any false positive.
  tau (main threshold): above tau0, include every candidate scoring >= tau.
  tau_fallback: if no candidate clears tau but the entity isn't gated as a
      singleton, predict the single best candidate if it clears the lower
      tau_fallback threshold. Rescues single-true-match entities whose best
      score falls between tau0 and tau.

Setting tau0=0 and tau_fallback=tau recovers a plain single-threshold rule
as a special case, so this is never worse than that baseline.

Also enforces the one VERIFIED hard constraint in the real ground truth
(checked directly against train_ground_truth.tsv: zero S2/S3 entity is ever
matched by more than one Source 1 entity across 7.6M distinct matched IDs):
a **greedy conflict-resolution pass** keeps only the highest-scoring edge
when the same candidate is predicted for more than one Source 1 entity.

Performance note: `sweep_thresholds` evaluates ~500+ (tau0, tau, tau_fallback)
combinations. The public `apply_thresholds`/`resolve_conflicts` functions
rebuild their pandas groupby/index structures from scratch on every call,
which is fine for a one-off call but was measurably wasteful (112s in an
early full smoke test) when called ~500x in a row on data that doesn't
change between iterations -- only the thresholds do. `sweep_thresholds`
therefore precomputes the per-entity sorted candidate lists and the
(entity, candidate) -> prob lookup ONCE and reuses them across the whole
grid, rather than calling the public DataFrame-based functions in the loop.
"""
from __future__ import annotations

import pandas as pd


def _precompute_entity_groups(pair_scores: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """source1_entity_id -> [(candidate_entity_id, prob), ...] sorted by prob descending."""
    groups: dict[str, list[tuple[str, float]]] = {}
    for row in pair_scores.itertuples(index=False):
        groups.setdefault(row.source1_entity_id, []).append((row.candidate_entity_id, row.prob))
    for pairs in groups.values():
        pairs.sort(key=lambda x: -x[1])
    return groups


def _apply_thresholds_precomputed(
    groups: dict[str, list[tuple[str, float]]],
    all_s1_ids: list[str],
    tau0: float,
    tau: float,
    tau_fallback: float,
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for eid in all_s1_ids:
        pairs = groups.get(eid)
        if not pairs:
            result[eid] = set()
            continue
        max_prob = pairs[0][1]  # already sorted descending
        if max_prob < tau0:
            result[eid] = set()
            continue
        above_tau = {cid for cid, p in pairs if p >= tau}
        if above_tau:
            result[eid] = above_tau
        elif max_prob >= tau_fallback:
            result[eid] = {pairs[0][0]}
        else:
            result[eid] = set()
    return result


def _resolve_conflicts_precomputed(
    pred_matches: dict[str, set[str]], score_lookup: dict[tuple[str, str], float]
) -> dict[str, set[str]]:
    candidate_to_entities: dict[str, list[str]] = {}
    for s1_id, cand_ids in pred_matches.items():
        for cid in cand_ids:
            candidate_to_entities.setdefault(cid, []).append(s1_id)

    result = {s1_id: set(cands) for s1_id, cands in pred_matches.items()}
    for cid, s1_ids in candidate_to_entities.items():
        if len(s1_ids) <= 1:
            continue
        best_s1 = max(s1_ids, key=lambda eid: score_lookup.get((eid, cid), 0.0))
        for eid in s1_ids:
            if eid != best_s1:
                result[eid].discard(cid)
    return result


def apply_thresholds(
    pair_scores: pd.DataFrame,  # columns: source1_entity_id, candidate_entity_id, prob
    all_s1_ids: list[str],
    tau0: float,
    tau: float,
    tau_fallback: float,
) -> dict[str, set[str]]:
    """Per-entity decision rule. `all_s1_ids` must include every entity that
    needs a row in the final output, even ones with zero candidates.

    One-shot public API -- for sweeping many threshold combinations over the
    same `pair_scores`, use `sweep_thresholds` instead, which precomputes the
    grouping once rather than paying this cost on every combination.
    """
    if pair_scores.empty:
        return {eid: set() for eid in all_s1_ids}
    groups = _precompute_entity_groups(pair_scores)
    return _apply_thresholds_precomputed(groups, all_s1_ids, tau0, tau, tau_fallback)


def resolve_conflicts(pred_matches: dict[str, set[str]], pair_scores: pd.DataFrame) -> dict[str, set[str]]:
    """Greedy conflict resolution: if the same candidate_entity_id is
    predicted for more than one Source 1 entity, keep only the
    highest-probability edge and drop it from the others (verified hard
    constraint -- see module docstring)."""
    score_lookup = dict(
        zip(zip(pair_scores["source1_entity_id"], pair_scores["candidate_entity_id"]), pair_scores["prob"])
    )
    return _resolve_conflicts_precomputed(pred_matches, score_lookup)


def sweep_thresholds(
    pair_scores: pd.DataFrame,
    true_matches: dict[str, set[str]],
    all_s1_ids: list[str],
    grid: list[float] | None = None,
    apply_conflict_resolution: bool = True,
) -> tuple[float, float, float, float]:
    """Grid search (tau0, tau, tau_fallback) maximizing real macro F0.5.
    Returns (best_tau0, best_tau, best_tau_fallback, best_score).
    Constraint enforced during the search: tau0 <= tau_fallback <= tau.
    """
    from . import evaluate  # local import to avoid a cycle at module load time

    if grid is None:
        grid = [round(x, 2) for x in [i / 100 for i in range(30, 96, 5)]]

    groups = _precompute_entity_groups(pair_scores)
    score_lookup = dict(
        zip(zip(pair_scores["source1_entity_id"], pair_scores["candidate_entity_id"]), pair_scores["prob"])
    )

    best = (0.0, grid[-1], 0.0, -1.0)
    for tau0 in grid:
        for tau_fallback in [g for g in grid if g >= tau0]:
            for tau in [g for g in grid if g >= tau_fallback]:
                preds = _apply_thresholds_precomputed(groups, all_s1_ids, tau0, tau, tau_fallback)
                if apply_conflict_resolution:
                    preds = _resolve_conflicts_precomputed(preds, score_lookup)
                score = evaluate.macro_f_beta(true_matches, preds, all_s1_ids)
                if score > best[3]:
                    best = (tau0, tau, tau_fallback, score)
    return best
