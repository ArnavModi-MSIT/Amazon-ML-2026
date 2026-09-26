"""v2/v3 pipeline: IDF top-K blocking -> (optional) relative-floor pruning ->
pairwise features + blocking, pool-frequency and within-entity features ->
LightGBM -> single threshold -> global one-to-one conflict resolution.

Design decisions, each from a real-scale measurement:
  - Blocking: blocking_idf (recall ceiling 0.33 -> ~0.94 at K=10/source).
  - Train on candidates generated against the FULL train pools with the exact
    inference candidate policy (sampled pools overstated precision; real pools
    are ~75% genuine matches of other S1 entities).
  - Pruning ("keep the top `min_rank`, plus any candidate scoring >= `rel_floor`
    x the entity's best") gives the best F0.5-per-candidate trade-off found:
    0.932 at 12.3 candidates/entity vs 0.934 at 20 (K=10), on held-out entities.
    It runs before feature computation and is applied identically in training
    and inference, so candidate_pairs.tsv is exactly what the model scores.
  - Pool frequency of the name/address keys: an exact name match means much
    more for a rare name than for a chain; string similarity alone can't show it.
  - Within-entity relative features: whether a candidate is the entity's best
    (by blocking score / name / address similarity) is precision evidence the
    per-pair features can't see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import blocking, blocking_idf, features, progress
from .parallel_utils import DEFAULT_N_JOBS, parallel_map

BLOCK_FEATURES = ["block_score", "block_rank", "block_top_score", "block_score_gap"]
FREQ_FEATURES = ["log_cand_name_key_freq", "log_cand_addr_key_freq", "log_s1_name_key_pool_freq",
                 "name_freq_x_addr_missing"]
LIST_FEATURES = ["block_ratio", "block_gap_top2", "n_within_90", "name_sort_rel", "name_lev_rel", "addr_tok_rel"]
FEATURE_NAMES_V3 = list(features.FEATURE_NAMES) + BLOCK_FEATURES + FREQ_FEATURES + LIST_FEATURES


def _key_freqs(rep: pd.DataFrame, col: str, s1_rep: pd.DataFrame | None = None):
    """Per-record pool frequency of `col` (0 for empty keys), as a Series
    indexed by the pool's entity_id -- and, if `s1_rep` is given, the pool
    frequency of each S1 record's own key, indexed by S1 entity_id."""
    keys = rep["country"].astype(str) + "|" + rep[col].astype(str)
    counts = keys.value_counts()
    freq = keys.map(counts).where(rep[col] != "", 0).astype(np.float32)
    pool_freq = pd.Series(freq.to_numpy(), index=rep["entity_id"].to_numpy())
    if s1_rep is None:
        return pool_freq
    s1_keys = s1_rep["country"].astype(str) + "|" + s1_rep[col].astype(str)
    s1_freq = s1_keys.map(counts).fillna(0).where(s1_rep[col] != "", 0).astype(np.float32)
    return pool_freq, pd.Series(s1_freq.to_numpy(), index=s1_rep["entity_id"].to_numpy())


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    k: int = blocking_idf.DEFAULT_K,
    max_df: int = blocking_idf.DEFAULT_MAX_DF,
    n_jobs: int = DEFAULT_N_JOBS,
    pool=None,
) -> pd.DataFrame:
    """All candidate pairs (both pools) with blocking scores and pool
    key-frequency columns."""
    progress.log(f"[v2] blocking reps: S1 {len(s1_df)}, S2 {len(s2_df)}, S3 {len(s3_df)}")
    s1_rep = blocking.add_blocking_representations(s1_df, n_jobs, pool=pool)
    parts = []
    for label, other in (("S2", s2_df), ("S3", s3_df)):
        rep = blocking.add_blocking_representations(other, n_jobs, pool=pool)
        progress.log(f"[v2] IDF top-{k} vs {label} (max_df={max_df})...")
        c = blocking_idf.topk_candidates(s1_rep, rep, k=k, max_df=max_df, label=f"-{label}")
        pool_nfreq, s1_nfreq = _key_freqs(rep, "name_key", s1_rep)
        pool_afreq = _key_freqs(rep, "addr_key")
        c["cand_name_key_freq"] = pool_nfreq.reindex(c["candidate_entity_id"]).to_numpy()
        c["cand_addr_key_freq"] = pool_afreq.reindex(c["candidate_entity_id"]).to_numpy()
        c["s1_name_key_pool_freq"] = s1_nfreq.reindex(c["source1_entity_id"]).to_numpy()
        c["source"] = label
        parts.append(c)
        del rep, pool_nfreq, s1_nfreq, pool_afreq
    cands = pd.concat(parts, ignore_index=True)
    progress.log(f"[v2] candidates: {len(cands)} pairs ({len(cands) / max(len(s1_df), 1):.2f}/entity)")
    return cands


def prune_candidates(cands: pd.DataFrame, rel_floor: float | None, min_rank: int) -> pd.DataFrame:
    """Keep ranks < min_rank, plus any candidate with block_score >=
    rel_floor * block_top_score (block_top_score is per entity per pool)."""
    if rel_floor is None:
        return cands
    keep = (cands["block_rank"] < min_rank) | (cands["block_score"] >= rel_floor * cands["block_top_score"])
    out = cands[keep].reset_index(drop=True)
    progress.log(f"[v2] pruned to {len(out)} pairs (floor {rel_floor}x top, min rank {min_rank})")
    return out


def featurize(
    cands: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    n_jobs: int = DEFAULT_N_JOBS,
    pool=None,
) -> pd.DataFrame:
    """Feature matrix (all feature columns) aligned row-for-row with `cands`.
    Callers select the columns their model was trained on."""
    need_s1 = set(cands["source1_entity_id"])
    need_other = set(cands["candidate_entity_id"])
    s1_full = blocking.add_representations(s1_df[s1_df["entity_id"].isin(need_s1)], n_jobs, pool=pool)
    other_full = pd.concat(
        [
            blocking.add_representations(s2_df[s2_df["entity_id"].isin(need_other)], n_jobs, pool=pool),
            blocking.add_representations(s3_df[s3_df["entity_id"].isin(need_other)], n_jobs, pool=pool),
        ],
        ignore_index=True,
    )
    s1_lookup = s1_full.set_index("entity_id").to_dict(orient="index")
    other_lookup = other_full.set_index("entity_id").to_dict(orient="index")
    del s1_full, other_full
    args = [
        (s1_lookup[a], other_lookup[b], 0 if b.startswith("S2-") else 1)
        for a, b in zip(cands["source1_entity_id"], cands["candidate_entity_id"])
    ]
    del s1_lookup, other_lookup
    rows = parallel_map(features.build_pair_features_from_tuple, args, pool=pool, n_jobs=n_jobs)
    del args
    X = pd.DataFrame(rows, columns=features.FEATURE_NAMES, dtype=np.float32)

    X["block_score"] = cands["block_score"].to_numpy(np.float32)
    X["block_rank"] = cands["block_rank"].to_numpy(np.float32)
    X["block_top_score"] = cands["block_top_score"].to_numpy(np.float32)
    X["block_score_gap"] = X["block_top_score"] - X["block_score"]

    X["log_cand_name_key_freq"] = np.log1p(cands["cand_name_key_freq"].to_numpy(np.float32))
    X["log_cand_addr_key_freq"] = np.log1p(cands["cand_addr_key_freq"].to_numpy(np.float32))
    X["log_s1_name_key_pool_freq"] = np.log1p(cands["s1_name_key_pool_freq"].to_numpy(np.float32))
    X["name_freq_x_addr_missing"] = X["log_cand_name_key_freq"] * X["addr_missing_either"]

    grp = [cands["source1_entity_id"].to_numpy(), cands["source"].to_numpy()]
    g = X.groupby(grp, sort=False)
    X["block_ratio"] = X["block_score"] / (X["block_top_score"] + 1e-6)
    # rank 1 is always kept by pruning (min_rank >= 2), so this is the true runner-up
    second = X["block_score"].where(X["block_rank"] == 1).groupby(grp, sort=False).transform("max").fillna(0.0)
    X["block_gap_top2"] = X["block_top_score"] - second
    X["n_within_90"] = (X["block_score"] >= 0.9 * X["block_top_score"]).groupby(grp, sort=False).transform("sum")
    X["name_sort_rel"] = X["name_token_sort_ratio"] - g["name_token_sort_ratio"].transform("max")
    X["name_lev_rel"] = X["name_levenshtein_ratio"] - g["name_levenshtein_ratio"].transform("max")
    X["addr_tok_rel"] = X["addr_token_jaccard"] - g["addr_token_jaccard"].transform("max")
    return X.astype(np.float32)
