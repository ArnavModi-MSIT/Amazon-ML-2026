"""Supervised meta-blocking: the LAST candidate-generation step, before the matching model.

A small LightGBM sees only blocking-stage signals, which are already computed for every candidate pair and cost no
string comparisons: cosine score / rank / the entity's top score, the exact-key flag, the reverse-channel flags
(is this S1 the pool record's best S1 among the whole S1 file?), and name/address key frequencies over the pool and
the S1 file. It drops candidates that cannot be matches, so `candidate_pairs.tsv` (= exactly what the matching model
scores) shrinks from ~13 to a few pairs per S1 entity while keeping a fixed share of the true pairs (chosen on
out-of-fold training predictions). Standard technique ("supervised meta-blocking", Papadakis et al.); the organisers
rank a smaller candidate set per S1 entity higher.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POOL_CLIP, S1_CLIP = 8, 4  # same clipping as the matcher's count features (test pools are smaller than train's)

STAGE1_FEATURES = [
    "block_score", "block_rank", "block_top_score", "block_ratio", "block_gap", "key_channel",
    "rev_top1", "rev_other", "rev_best_ratio", "n_src_cands", "field_weighted", "is_s3",
    "log_cand_name_key_freq", "log_cand_addr_key_freq", "log_s1_name_key_pool_freq",
    "log_cand_core_s1_freq", "log_cand_core_pool_freq", "log_cand_addrkey_s1_freq", "log_s1_core_s1_freq",
    "log_s1_addrkey_s1_freq", "cand_tok_s1_frac", "log_cand_addrfull_s1_freq", "log_cand_addrfull_pool_freq",
    "log_cand_core_pool_naddr", "log_cand_addr_pool_nnames",
]
_POOL_COUNTS = ["cand_name_key_freq", "cand_addr_key_freq", "s1_name_key_pool_freq", "cand_core_pool_freq",
                "cand_addrfull_pool_freq", "cand_core_pool_naddr", "cand_addr_pool_nnames"]
_S1_COUNTS = ["cand_core_s1_freq", "cand_addrkey_s1_freq", "s1_core_s1_freq", "s1_addrkey_s1_freq",
              "cand_addrfull_s1_freq"]
PARAMS = {"objective": "binary", "learning_rate": 0.08, "num_leaves": 63, "min_child_samples": 100,
          "subsample": 0.8, "bagging_freq": 1, "colsample_bytree": 0.8, "verbose": -1, "seed": 7}
N_ROUNDS = 400


def stage1_matrix(cands: pd.DataFrame, country_of: pd.Series, weighted_countries) -> pd.DataFrame:
    """Blocking-stage feature matrix aligned with `cands` (output of generate_candidates + prune_candidates)."""
    X = pd.DataFrame(index=cands.index)
    s = cands["block_score"].to_numpy(np.float32)
    top = cands["block_top_score"].to_numpy(np.float32)
    X["block_score"], X["block_top_score"] = s, top
    X["block_rank"] = cands["block_rank"].to_numpy(np.float32)
    X["block_ratio"] = np.where(top > 0, s / np.maximum(top, 1e-6), 0).astype(np.float32)
    X["block_gap"] = (top - s).astype(np.float32)
    X["key_channel"] = cands["key_channel"].to_numpy(np.float32) if "key_channel" in cands else np.float32(0)
    for c in ("rev_top1", "rev_other"):
        X[c] = cands[c].to_numpy(np.float32) if c in cands else np.float32(0)
    best = cands["rev_best_score"].to_numpy(np.float32) if "rev_best_score" in cands else np.zeros(len(cands), np.float32)
    X["rev_best_ratio"] = np.where(best > 0, np.minimum(s / np.maximum(best, 1e-6), 2), 0).astype(np.float32)
    X["n_src_cands"] = cands.groupby(["source1_entity_id", "source"])["candidate_entity_id"].transform("size") \
        .to_numpy(np.float32)
    cty = cands["source1_entity_id"].map(country_of)
    X["field_weighted"] = cty.isin(set(weighted_countries or ())).to_numpy(np.float32)
    X["is_s3"] = (cands["source"] == "S3").to_numpy(np.float32)
    for c in _POOL_COUNTS + _S1_COUNTS:
        clip = POOL_CLIP if c in _POOL_COUNTS else S1_CLIP
        v = cands[c].to_numpy(np.float32) if c in cands else np.zeros(len(cands), np.float32)
        X[f"log_{c}"] = np.log1p(np.minimum(np.nan_to_num(v), clip)).astype(np.float32)
    X["cand_tok_s1_frac"] = cands["cand_tok_s1_frac"].to_numpy(np.float32) if "cand_tok_s1_frac" in cands \
        else np.float32(0)
    return X[STAGE1_FEATURES]


def train(X: pd.DataFrame, y: np.ndarray):
    import lightgbm as lgb
    return lgb.train(PARAMS, lgb.Dataset(X, y), N_ROUNDS)


def fit_with_oof(X: pd.DataFrame, y: np.ndarray, entity: pd.Series, is_fit: np.ndarray, keep_recall: float):
    """Two-fold out-of-fold probabilities on the fit rows (by entity) choose the threshold that keeps
    `keep_recall` of the true pairs; the returned model is trained on all fit rows."""
    ents = np.array(sorted(set(entity[is_fit])))
    fold_a = set(ents[::2])
    in_a = is_fit & entity.isin(fold_a).to_numpy()
    in_b = is_fit & ~in_a
    oof = np.zeros(len(X), np.float32)
    for trn, tst in ((in_b, in_a), (in_a, in_b)):
        oof[tst] = train(X[trn], y[trn]).predict(X[tst])
    pos = np.sort(oof[is_fit & (y == 1)])
    tau = float(pos[int(np.floor((1 - keep_recall) * len(pos)))])
    return train(X[is_fit], y[is_fit]), tau, oof
