"""Candidate generation: IDF-weighted word-token cosine, top-K per S1 entity.

Each record becomes a bag of word tokens -- name tokens (from both the
normalized name and its transliteration skeleton, so cross-script pairs can
share tokens), address tokens and, for native-script names, phonetic codes,
field-prefixed ("n:"/"a:"/"p:"; see `record_tokens`). Per country,
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

import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse

from . import progress, text_repr as tr

DEFAULT_K = 10
DEFAULT_MAX_DF = 50_000
MATMUL_BATCH = 100_000


def record_tokens(name_norm: str, name_skeleton: str, addr_norm: str, phonetic: bool, name_seg: str = "",
                  latin_only: bool = False) -> list[str]:
    """Bag of field-prefixed tokens:
      n: name words (normalized + transliteration skeleton), a: address words
         (any digit run counts -- single-digit house numbers are specific),
      p: phonetic codes of the skeleton words, so a native-script name meets
         its English spelling ("லக்ஷ்மி" and "Laxmi" -> p:425). Only S1 and
         non-ASCII pool names carry them (`phonetic`), which keeps the pool
         matrix small: S1 is all Latin, so Latin-Latin pairs already share
         n: tokens.
    Measured on the 20k held-out entities (pruned recall ceiling): new
    normalization 0.9326, + single-digit numbers 0.9330, + p: 0.9399. A
    whole-name "j:" token (to catch "visioncarelynn.com") HURT: 0.8976 -- a
    rare exact-name token pulls same-name decoys above the true duplicate,
    whose name is noisy -- so name joining is only a model feature.
    `name_seg`: S1-vocabulary segmentation of a glued pool name
    ("calkinxflh" -> "calkin xflh"), added as ordinary n: words."""
    # latin_only: drop non-ASCII words. S1 is entirely Latin, so a native-script word in a pool record can
    # never match a query; it only inflates that record's norm and ranks native-script records lower.
    live = str.isascii if latin_only else (lambda t: True)
    toks = {"n:" + t for t in name_norm.split() if len(t) >= 2 and live(t)}
    toks |= {"n:" + t for t in name_skeleton.split() if len(t) >= 2 and live(t)}
    toks |= {"a:" + t for t in addr_norm.split() if (len(t) >= 2 or t.isdigit()) and live(t)}
    if phonetic:
        toks |= {"p:" + c for c in tr.phonetic_tokens(name_skeleton)}
    if name_seg:
        toks |= {"n:" + t for t in name_seg.split() if len(t) >= 2}
    return list(toks)


# ---------------------------------------------------------------------------
# Glued-name segmentation. ~20% of v4's blocking misses are pool names that
# are one glued token ("visioncarelynn.com", "optroquetrustedsunshine") made
# entirely of the S1 entity's own words. Segmenting them into S1-vocabulary
# words gives ordinary word tokens with ordinary IDF (unlike the rejected
# whole-name j: token, records with normal names score exactly as before).
# The vocabulary is the S1 file of the same split, so the true entity's own
# words are always in it; no external dictionary.
# ---------------------------------------------------------------------------
SEG_MIN_LEN = 7        # only glued tokens at least this long
SEG_MAX_WORD = 20
SEG_MAX_PARTS = 4
SEG_SINGLE_COST = 12.0  # allow one initial: "everettmmargery" -> everett m margery


def build_segment_costs(s1_rep: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per country: S1 name word -> -log(relative frequency)."""
    out = {}
    for cty, names in s1_rep.groupby("country")["name_norm"]:
        cnt = Counter(t for n in names for t in n.split() if len(t) >= 2 and t.isascii() and t.isalpha())
        total = sum(cnt.values())
        out[str(cty)] = {w: math.log(total / c) for w, c in cnt.items()}
    return out


def _segment(tok: str, costs: dict[str, float]) -> str:
    n, inf = len(tok), float("inf")
    best, back = [0.0] + [inf] * n, [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - SEG_MAX_WORD), i):
            if best[j] == inf:
                continue
            c = costs.get(tok[j:i])
            if c is None:
                if i - j != 1:
                    continue
                c = SEG_SINGLE_COST
            if best[j] + c < best[i]:
                best[i], back[i] = best[j] + c, j
    if best[n] == inf:
        return ""
    parts, i = [], n
    while i:
        parts.append(tok[back[i]:i])
        i = back[i]
    parts.reverse()
    if not 2 <= len(parts) <= SEG_MAX_PARTS or sum(len(p) == 1 for p in parts) > 1:
        return ""
    return " ".join(parts)


def segment_names(rep: pd.DataFrame, costs: dict[str, dict[str, float]]) -> list[str]:
    """Segmentation of glued single-token names ('' for every other record)."""
    out, cache = [], {}
    for cty, name in zip(rep["country"].astype(str), rep["name_norm"]):
        toks = [t for t in name.split() if t not in tr.LEGAL_SUFFIXES]
        seg = ""
        if len(toks) == 1 and len(toks[0]) >= SEG_MIN_LEN and toks[0].isascii() and toks[0].isalpha():
            c = costs.get(cty)
            if c is not None and toks[0] not in c:
                key = (cty, toks[0])
                if key not in cache:
                    cache[key] = _segment(toks[0], c)
                seg = cache[key]
        out.append(seg)
    return out


def _identity(doc):
    return doc


def _mask_and_normalize(mat: sparse.csr_matrix, keep: np.ndarray) -> sparse.csr_matrix:
    m = (mat @ sparse.diags(keep.astype(np.float32))).tocsr()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sparse.diags((1.0 / norms).astype(np.float32)) @ m).tocsr()


def _field_normalize(mat: sparse.csr_matrix, keep: np.ndarray, is_addr: np.ndarray, weights,
                     empty_addr_c: float | None = None) -> sparse.csr_matrix:
    """L2-normalize the name block and the address block of each row
    separately, then weight them: Q.P = w_name*cos_name + w_addr*cos_addr
    when `weights` is given on the pool side and (1, 1) on the query side.
    With one joint norm, a single very rare token (a native-script word,
    "calkinxflh.com") takes nearly all of a pool record's weight, so an
    exactly matching address barely counts.

    `empty_addr_c` (pool side): a pool row with no address tokens left after
    masking scores w_name*cos_name + c*w_addr*cos_name, i.e. its missing
    address counts as a partial agreement instead of zero -- otherwise every
    same-name decoy with any shared address token outranks an empty-address
    true duplicate."""
    w_name, w_addr = weights
    name = _mask_and_normalize(mat, keep & ~is_addr)
    addr = _mask_and_normalize(mat, keep & is_addr)
    if empty_addr_c is not None:
        no_addr = np.diff(addr.indptr) == 0
        scale = np.where(no_addr, 1.0 + empty_addr_c * w_addr / w_name, 1.0).astype(np.float32)
        name = sparse.diags(scale) @ name
    return (name * np.float32(w_name) + addr * np.float32(w_addr)).tocsr()


def topk_candidates(
    s1_rep: pd.DataFrame,
    pool_rep: pd.DataFrame,
    k: int = DEFAULT_K,
    max_df: int = DEFAULT_MAX_DF,
    label: str = "",
    field_weights: tuple[float, ...] | None = None,
    latin_only_tokens: bool = False,
) -> pd.DataFrame:
    """`s1_rep`/`pool_rep` need entity_id, country, name_norm, name_skeleton,
    addr_norm (e.g. from blocking.add_blocking_representations).

    Returns one row per candidate pair: source1_entity_id,
    candidate_entity_id, block_score (cosine), block_rank (0 = best within
    this S1 entity for this pool), block_top_score (that entity's best
    score in this pool).

    `field_weights=(w_name, w_addr[, c])`, or a {country: (...)} dict (other
    countries: joint cosine), scores w_name*cos(name tokens) +
    w_addr*cos(address tokens) instead of one joint cosine; `c` treats an
    empty pool address as partial agreement (see `_field_normalize`)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn

    out = []
    for country in sorted(set(s1_rep["country"]) & set(pool_rep["country"])):
        q = s1_rep[s1_rep["country"] == country]
        p = pool_rep[pool_rep["country"] == country]
        vec = TfidfVectorizer(analyzer=_identity, sublinear_tf=True, dtype=np.float32)
        P = vec.fit_transform(
            [record_tokens(a, b, c, not a.isascii(), d, latin_only_tokens)
             for a, b, c, d in zip(p["name_norm"], p["name_skeleton"], p["addr_norm"],
                                   p["name_seg"] if "name_seg" in p else [""] * len(p))]
        ).tocsr()
        keep = np.bincount(P.indices, minlength=P.shape[1]) <= max_df
        Q = vec.transform(
            [record_tokens(a, b, c, True, "", latin_only_tokens)
             for a, b, c in zip(q["name_norm"], q["name_skeleton"], q["addr_norm"])]
        ).tocsr()
        fw = field_weights.get(country) if isinstance(field_weights, dict) else field_weights
        if fw is None:
            PT = _mask_and_normalize(P, keep).T.tocsr()
            Q = _mask_and_normalize(Q, keep)
        else:
            is_addr = np.char.startswith(vec.get_feature_names_out().astype(str), "a:")
            w_name, w_addr, *c = fw  # optional third value: empty-address neutral cosine
            PT = _field_normalize(P, keep, is_addr, (w_name, w_addr), c[0] if c else None).T.tocsr()
            Q = _field_normalize(Q, keep, is_addr, (1.0, 1.0))
        del P
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


def reverse_topk(
    s1_rep: pd.DataFrame,
    pool_rep: pd.DataFrame,
    k: int = 2,
    max_df: int = DEFAULT_MAX_DF,
    label: str = "",
    field_weights=None,
    latin_only_tokens: bool = False,
) -> pd.DataFrame:
    """Reverse direction of `topk_candidates`: for every POOL record, its top-k S1 records under the same
    cosine (same vectorizer fit on the pool, same token policy and normalization). Each pool record belongs
    to at most one S1 entity, so the S1 entity it scores highest is a natural candidate even when that
    entity's own forward top-K is crowded by look-alikes. `s1_rep` must be the WHOLE S1 file of the split,
    so an entity competes with every other S1 record. Idea from a public competition repo (reverse pass).
    Returns source1_entity_id, candidate_entity_id, rev_score, rev_rank (0 = the pool record's best S1)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn

    out = []
    for country in sorted(set(s1_rep["country"]) & set(pool_rep["country"])):
        q = s1_rep[s1_rep["country"] == country]
        p = pool_rep[pool_rep["country"] == country]
        vec = TfidfVectorizer(analyzer=_identity, sublinear_tf=True, dtype=np.float32)
        P = vec.fit_transform(
            [record_tokens(a, b, c, not a.isascii(), d, latin_only_tokens)
             for a, b, c, d in zip(p["name_norm"], p["name_skeleton"], p["addr_norm"],
                                   p["name_seg"] if "name_seg" in p else [""] * len(p))]
        ).tocsr()
        keep = np.bincount(P.indices, minlength=P.shape[1]) <= max_df
        Q = vec.transform(
            [record_tokens(a, b, c, True, "", latin_only_tokens)
             for a, b, c in zip(q["name_norm"], q["name_skeleton"], q["addr_norm"])]
        ).tocsr()
        fw = field_weights.get(country) if isinstance(field_weights, dict) else field_weights
        if fw is None:
            Pn, Qn = _mask_and_normalize(P, keep), _mask_and_normalize(Q, keep)
        else:
            is_addr = np.char.startswith(vec.get_feature_names_out().astype(str), "a:")
            w_name, w_addr, *c = fw
            Pn = _field_normalize(P, keep, is_addr, (w_name, w_addr), c[0] if c else None)
            Qn = _field_normalize(Q, keep, is_addr, (1.0, 1.0))
        del P, Q
        QT = Qn.T.tocsr()
        del Qn
        q_ids, p_ids = q["entity_id"].to_numpy(), p["entity_id"].to_numpy()
        progress.log(f"  [rev{label}] {country}: {len(p)} pool x {len(q)} S1")
        for start in range(0, Pn.shape[0], MATMUL_BATCH):
            S = sp_matmul_topn(Pn[start:start + MATMUL_BATCH], QT, top_n=k, threshold=0.0, n_threads=-1).tocsr()
            rows = np.repeat(np.arange(S.shape[0]), np.diff(S.indptr))
            out.append(pd.DataFrame({
                "source1_entity_id": q_ids[S.indices],
                "candidate_entity_id": p_ids[start + rows],
                "rev_score": S.data.astype(np.float32),
            }))
        del Pn, QT
    res = pd.concat(out, ignore_index=True)
    res["rev_rank"] = (res.groupby("candidate_entity_id")["rev_score"]
                       .rank(method="first", ascending=False).astype(np.int16) - 1)
    return res
