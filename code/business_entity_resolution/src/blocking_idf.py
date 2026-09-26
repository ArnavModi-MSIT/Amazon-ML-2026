"""Candidate generation: IDF-weighted word-token cosine, top-K per S1 entity.

Each record becomes a bag of word tokens -- name tokens (from both the
normalized name and its transliteration skeleton, so cross-script pairs can
share tokens) and address tokens, field-prefixed ("n:"/"a:"). Per country,
a TF-IDF model is fit on the pool (S2 or S3), tokens with document
frequency above `max_df` are dropped (they carry almost no weight and
dominate compute), and for every S1 record the K pool records with the
highest cosine similarity become its candidates.

Why this design (measured on 20k held-out train entities vs the FULL train
pools): ~100% of true pairs share at least one word token, but shared
tokens are often common, so absolute frequency caps trade recall for huge
volume (the previous Splink key/token blocking reached only 0.33 recall).
Ranking by IDF-weighted overlap lets rare shared tokens dominate while K
bounds the candidate set: S2 recall@10 = 0.94 at max_df=50k (0.915 @5).

Country is an exact partition: train ground truth has zero cross-country
matches in 7.6M pairs. Countries are taken from the data (open set), so
test-only France is handled like any other country.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from . import progress

DEFAULT_K = 10
DEFAULT_MAX_DF = 50_000
MATMUL_BATCH = 100_000


def record_tokens(name_norm: str, name_skeleton: str, addr_norm: str) -> list[str]:
    toks = {"n:" + t for t in name_norm.split() if len(t) >= 2}
    toks |= {"n:" + t for t in name_skeleton.split() if len(t) >= 2}
    toks |= {"a:" + t for t in addr_norm.split() if len(t) >= 2}
    return list(toks)


def _identity(doc):
    return doc


def _mask_and_normalize(mat: sparse.csr_matrix, keep: np.ndarray) -> sparse.csr_matrix:
    m = (mat @ sparse.diags(keep.astype(np.float32))).tocsr()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sparse.diags((1.0 / norms).astype(np.float32)) @ m).tocsr()


def topk_candidates(
    s1_rep: pd.DataFrame,
    pool_rep: pd.DataFrame,
    k: int = DEFAULT_K,
    max_df: int = DEFAULT_MAX_DF,
    label: str = "",
) -> pd.DataFrame:
    """`s1_rep`/`pool_rep` need entity_id, country, name_norm, name_skeleton,
    addr_norm (e.g. from blocking.add_blocking_representations).

    Returns one row per candidate pair: source1_entity_id,
    candidate_entity_id, block_score (cosine), block_rank (0 = best within
    this S1 entity for this pool), block_top_score (that entity's best
    score in this pool)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn

    out = []
    for country in sorted(set(s1_rep["country"]) & set(pool_rep["country"])):
        q = s1_rep[s1_rep["country"] == country]
        p = pool_rep[pool_rep["country"] == country]
        vec = TfidfVectorizer(analyzer=_identity, sublinear_tf=True, dtype=np.float32)
        P = vec.fit_transform(
            [record_tokens(a, b, c) for a, b, c in zip(p["name_norm"], p["name_skeleton"], p["addr_norm"])]
        ).tocsr()
        keep = np.bincount(P.indices, minlength=P.shape[1]) <= max_df
        PT = _mask_and_normalize(P, keep).T.tocsr()
        del P
        Q = _mask_and_normalize(
            vec.transform(
                [record_tokens(a, b, c) for a, b, c in zip(q["name_norm"], q["name_skeleton"], q["addr_norm"])]
            ).tocsr(),
            keep,
        )
        q_ids = q["entity_id"].to_numpy()
        p_ids = p["entity_id"].to_numpy()
        progress.log(f"  [idf{label}] {country}: {len(q)} S1 x {len(p)} pool, vocab kept {int(keep.sum())}")
        for start in range(0, Q.shape[0], MATMUL_BATCH):
            S = sp_matmul_topn(Q[start:start + MATMUL_BATCH], PT, top_n=k, threshold=0.0, n_threads=-1).tocsr()
            rows = np.repeat(np.arange(S.shape[0]), np.diff(S.indptr))
            df = pd.DataFrame({
                "source1_entity_id": q_ids[start + rows],
                "candidate_entity_id": p_ids[S.indices],
                "block_score": S.data.astype(np.float32),
            })
            out.append(df)
        del PT, Q
    if not out:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "block_score",
                                     "block_rank", "block_top_score"])
    res = pd.concat(out, ignore_index=True)
    res["block_rank"] = (
        res.groupby("source1_entity_id")["block_score"].rank(method="first", ascending=False).astype(np.int16) - 1
    )
    res["block_top_score"] = res.groupby("source1_entity_id")["block_score"].transform("max").astype(np.float32)
    return res
