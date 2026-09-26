"""Phase 3 -- Multi-channel candidate generation (blocking).

Builds a candidate set of (source1_entity_id -> {candidate_entity_ids}) using
several cheap channels unioned together, per the plan's reconciliation of
four external reviews:

  1. Character n-gram TF-IDF cosine similarity (primary fuzzy channel) --
     fit *per country* (not globally) so high-frequency English tokens don't
     drown out India/France signal, computed with `sparse_dot_topn` for a
     vectorized top-K-per-row sparse matrix multiply instead of a Python
     double loop.
  2. Exact normalized-name key (cheap, catches exact/near-exact matches the
     fuzzy channel might not rank in its top-K).
  3. Exact composite address key (leading street-number digits + first
     address token) -- catches cases where the name is heavily
     transliterated but the address is cleaner.

Frequency control: channel 2/3 posting lists longer than `freq_cap` are
skipped entirely for that key (a common chain name must not explode into a
massive candidate set) -- those records still get covered by channel 1.

Empty-address guard: never emit an address-based key for an empty address
(would otherwise create one giant false-candidate bucket linking every
empty-address record together).

KNOWN LIMITATION (documented, not silently ignored): country partitioning is
a *soft* signal per the plan, but this implementation partitions by country
for tractability -- a genuine match with inconsistent/wrong country labeling
between sources would be missed here. Phase 3b's per-country candidate-recall
measurement is what will reveal whether this matters enough to add a
lower-priority cross-country fallback channel; not built preemptively per
the reviews' "measurement over intuition" principle.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd

from . import progress, text_repr as tr
from .parallel_utils import DEFAULT_N_JOBS, parallel_map, parallel_pool

DEFAULT_TOP_K = 15
DEFAULT_FREQ_CAP = 500  # max posting-list size for an exact-key channel
MIN_NGRAM_DF = 1  # per-country buckets are small enough that df>=2 can be too strict


def _compute_reprs(df: pd.DataFrame, n_jobs: int, pool):
    """Shared step: run text_repr over every row via the process pool.
    Returns (name_reprs, addr_reprs) lists of dataclass objects -- callers
    extract only the fields they need and should `del` these lists promptly
    once done (they're large: one object per row)."""
    name_reprs = parallel_map(tr.build_name_repr, df["business_name"].tolist(), pool=pool, n_jobs=n_jobs)
    addr_reprs = parallel_map(tr.build_address_repr_pair,
                              list(zip(df["business_address"].tolist(), df["country"].astype(str).tolist())),
                              pool=pool, n_jobs=n_jobs)
    return name_reprs, addr_reprs


def add_blocking_representations(
    df: pd.DataFrame, n_jobs: int = DEFAULT_N_JOBS, pool=None
) -> pd.DataFrame:
    """Lightweight representation with ONLY the columns blocking's exact-key
    and TF-IDF channels actually read: entity_id, country, name_norm,
    name_skeleton, addr_norm, name_key, addr_key. Does NOT include tokens,
    suffix flags, dominant_script, or digit tokens -- those are only read by
    `features.py`, for the much smaller set of records that survive blocking
    as candidates, not the full pool.

    Found necessary after a real full-scale run: computing and storing the
    FULL representation (10 derived columns, see `add_representations`) for
    the entire ~10M-row S2/S3 pool pushed memory from ~3.5GB to ~9.5GB for
    S1 alone (1.7M rows) and was trending toward exhausting 32GB before S2/S3
    (5M+ rows each) even finished -- most of that memory paid for columns
    blocking never reads. The fix: block using this cheap representation for
    the FULL pool, then compute the expensive full representation (below)
    only for the filtered-down candidate subset afterward.
    """
    name_reprs, addr_reprs = _compute_reprs(df, n_jobs, pool)
    name_norm = [r.normalized for r in name_reprs]
    name_skeleton = [r.skeleton for r in name_reprs]
    addr_norm = [r.normalized for r in addr_reprs]
    addr_leading_digits = [r.leading_digits for r in addr_reprs]
    addr_first_token = [r.first_word for r in addr_reprs]
    del name_reprs, addr_reprs  # free promptly -- see _compute_reprs docstring

    name_norm_s = pd.Series(name_norm, index=df.index)
    name_skeleton_s = pd.Series(name_skeleton, index=df.index)
    addr_leading_digits_s = pd.Series(addr_leading_digits, index=df.index)
    addr_first_token_s = pd.Series(addr_first_token, index=df.index)

    out = pd.DataFrame(
        {
            "entity_id": df["entity_id"].to_numpy(),
            "country": df["country"].to_numpy(),
            "name_norm": name_norm_s,
            "name_skeleton": name_skeleton_s,
            "addr_norm": addr_norm,
            "name_key": name_skeleton_s.where(name_skeleton_s != "", name_norm_s),
            "addr_key": np.where(
                (addr_leading_digits_s != "") & (addr_first_token_s != ""),
                addr_leading_digits_s + "|" + addr_first_token_s,
                "",  # empty address / no leading digit -> no key (empty-address guard)
            ),
        }
    )
    return out


def add_representations(
    df: pd.DataFrame, n_jobs: int = DEFAULT_N_JOBS, pool=None
) -> pd.DataFrame:
    """Add ALL derived columns (normalized/skeleton/key + tokens/suffix/
    script/digit-tokens needed by `features.py`). Only call this on records
    that actually need full features computed -- for blocking, use the much
    cheaper `add_blocking_representations` instead (see its docstring for why
    this distinction matters at scale).

    This was the biggest single-core bottleneck in the pipeline (regex +
    Unicode work per record, done in a plain Python loop) -- now spread
    across a process pool via `parallel_map`.

    Pass `pool=` (from `parallel_utils.parallel_pool()`) when calling this
    multiple times in the same pipeline run so all calls share one pool
    instead of each one spawning its own -- see parallel_utils.py's module
    docstring for why creating a fresh pool per call caused a real
    system-wide freeze.
    """
    df = df.copy()
    name_reprs, addr_reprs = _compute_reprs(df, n_jobs, pool)

    df["name_norm"] = [r.normalized for r in name_reprs]
    df["name_skeleton"] = [r.skeleton for r in name_reprs]
    df["name_tokens_no_suffix"] = [tuple(r.tokens_no_suffix) for r in name_reprs]
    df["name_suffix_tokens"] = [tuple(r.suffix_tokens) for r in name_reprs]
    df["dominant_script"] = [r.dominant_script for r in name_reprs]

    df["addr_norm"] = [r.normalized for r in addr_reprs]
    df["addr_leading_digits"] = [r.leading_digits for r in addr_reprs]
    df["addr_first_token"] = [r.first_word for r in addr_reprs]
    df["addr_digit_tokens"] = [tuple(r.digit_tokens) for r in addr_reprs]
    df["addr_postcodes"] = [r.postcodes for r in addr_reprs]
    del name_reprs, addr_reprs  # free promptly -- see _compute_reprs docstring

    # Composite blocking keys -- combine skeleton (transliteration-tolerant)
    # with the original normalized name so exact-script and cross-script
    # exact matches are both caught by the same exact-key channel.
    df["name_key"] = df["name_skeleton"].where(df["name_skeleton"] != "", df["name_norm"])
    df["addr_key"] = np.where(
        (df["addr_leading_digits"] != "") & (df["addr_first_token"] != ""),
        df["addr_leading_digits"] + "|" + df["addr_first_token"],
        "",  # empty address / no leading digit -> no key (empty-address guard)
    )
    return df


def _exact_key_candidates(
    s1_df: pd.DataFrame, other_df: pd.DataFrame, key_col: str, freq_cap: int
) -> dict[str, set[str]]:
    """Bucket both frames by `key_col`, union entity_ids sharing a non-empty
    key, skip (cap) any bucket whose *other*-side posting list is too long.
    """
    other_groups = other_df.groupby(key_col)["entity_id"].apply(list)
    result: dict[str, set[str]] = defaultdict(set)
    s1_by_key = s1_df.groupby(key_col)["entity_id"].apply(list)
    for key, s1_ids in s1_by_key.items():
        if not key:
            continue
        other_ids = other_groups.get(key)
        if not other_ids or len(other_ids) > freq_cap:
            continue  # frequency control: skip pathologically common keys
        for s1_id in s1_ids:
            result[s1_id].update(other_ids)
    return result


def _tfidf_topk_candidates(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    text_col: str,
    top_k: int,
    threshold: float = 0.1,
) -> dict[str, set[str]]:
    """Char n-gram TF-IDF cosine top-K per S1 row, partitioned by country.

    Fits one vectorizer per country bucket present in *this* call's data
    (transductive -- allowed, no external data), so India/France records
    aren't drowned out by high-frequency English tokens from a global fit.

    sklearn/sparse_dot_topn are imported lazily, inside this function, not at
    module level. Reason: worker processes spawned for `add_representations`
    or feature computation only ever call `text_repr`/`features` functions,
    never this one -- but under Windows `spawn`, a worker re-imports this
    whole module anyway just to reconstruct the __main__ import graph. A
    module-level sklearn import means every worker pays the full
    scipy/sklearn import chain (the exact DLL-load chain that caused the
    page-file-exhaustion freeze) for nothing. Deferring it here means workers
    only pay for pandas/numpy -- lighter, and skips scipy/sklearn entirely.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn

    result: dict[str, set[str]] = defaultdict(set)
    for country, s1_group in s1_df.groupby("country"):
        other_group = other_df[other_df["country"] == country]
        if other_group.empty or s1_group.empty:
            continue
        progress.log(
            f"    tfidf[{text_col}] country={country}: {len(s1_group)} S1 rows vs {len(other_group)} other rows"
        )
        s1_texts = s1_group[text_col].tolist()
        other_texts = other_group[text_col].tolist()
        # Guard: an all-empty-string bucket has no signal to vectorize.
        if not any(s1_texts) or not any(other_texts):
            continue
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=MIN_NGRAM_DF, sublinear_tf=True
        )
        # Fit on the union so both sides share one vocabulary/IDF weighting.
        vectorizer.fit(s1_texts + other_texts)
        s1_mat = vectorizer.transform(s1_texts).astype(np.float32)
        other_mat = vectorizer.transform(other_texts).astype(np.float32)
        if s1_mat.nnz == 0 or other_mat.nnz == 0:
            continue
        sims = sp_matmul_topn(
            s1_mat, other_mat.T.tocsr(), top_n=top_k, threshold=threshold, n_threads=-1
        )
        sims = sims.tocsr()
        s1_ids = s1_group["entity_id"].to_numpy()
        other_ids = other_group["entity_id"].to_numpy()
        for row_idx in range(sims.shape[0]):
            start, end = sims.indptr[row_idx], sims.indptr[row_idx + 1]
            if start == end:
                continue
            cols = sims.indices[start:end]
            result[s1_ids[row_idx]].update(other_ids[cols])
    return result


def generate_candidates_for_source(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    freq_cap: int = DEFAULT_FREQ_CAP,
    tfidf_threshold: float = 0.1,
    verbose: bool = True,
) -> dict[str, set[str]]:
    """Union of all blocking channels for one Source1-vs-other-source pair.

    `s1_df`/`other_df` must already have the representation columns added by
    `add_representations`. `tfidf_threshold` and `top_k` directly control
    candidate volume -- raise `tfidf_threshold` / lower `top_k` to trade
    recall for a smaller, faster-to-process candidate set (see the
    full-scale timing note in plan.md: at the defaults, measured
    candidates/entity averaged ~124, far above the ~15-20 the top_k alone
    would suggest, because five channels are unioned together).
    """
    channels = {}
    t0 = time.time()
    progress.log(f"  channel name_key: starting ({len(s1_df)} S1 rows, {len(other_df)} other rows)")
    channels["name_key"] = _exact_key_candidates(s1_df, other_df, "name_key", freq_cap)
    progress.log("  channel addr_key: starting")
    channels["addr_key"] = _exact_key_candidates(s1_df, other_df, "addr_key", freq_cap)
    progress.log("  channel tfidf_name: starting")
    channels["tfidf_name"] = _tfidf_topk_candidates(s1_df, other_df, "name_norm", top_k, tfidf_threshold)
    # Critical fix (found via smoke-test diagnosis on real data): a
    # transliterated name (e.g. Devanagari) shares zero character n-grams
    # with its Latin counterpart in `name_norm`, so the fuzzy channel above
    # scores those pairs at 0 similarity even though the exact-key skeleton
    # channel might catch some of them. Running TF-IDF on the skeleton
    # representation too directly targets the cross-script case.
    progress.log("  channel tfidf_name_skeleton: starting")
    channels["tfidf_name_skeleton"] = _tfidf_topk_candidates(
        s1_df, other_df, "name_skeleton", top_k, tfidf_threshold
    )
    # Second fix from the same diagnosis: addresses with token reordering and
    # no leading street number (common in India addresses starting with an
    # area/locality name) were invisible to the exact addr_key channel.
    progress.log("  channel tfidf_addr: starting")
    channels["tfidf_addr"] = _tfidf_topk_candidates(s1_df, other_df, "addr_norm", top_k, tfidf_threshold)
    if verbose:
        elapsed = time.time() - t0
        for name, d in channels.items():
            n_covered = sum(1 for v in d.values() if v)
            progress.log(f"  channel {name}: {n_covered} S1 entities with >=1 candidate "
                         f"({elapsed:.1f}s cumulative)")

    merged: dict[str, set[str]] = defaultdict(set)
    for channel_result in channels.values():
        for s1_id, cand_ids in channel_result.items():
            merged[s1_id].update(cand_ids)
    return merged


class PoolIndex:
    """Precomputed, S1-independent index over one "other" source's pool
    (S2 or S3), built ONCE and reused across every S1 batch in a batched
    inference run -- see `build_pool_index` for why this exists."""

    __slots__ = ("name_key_groups", "addr_key_groups", "tfidf_name", "tfidf_name_skeleton", "tfidf_addr")

    def __init__(self, name_key_groups, addr_key_groups, tfidf_name, tfidf_name_skeleton, tfidf_addr):
        self.name_key_groups = name_key_groups
        self.addr_key_groups = addr_key_groups
        self.tfidf_name = tfidf_name
        self.tfidf_name_skeleton = tfidf_name_skeleton
        self.tfidf_addr = tfidf_addr


def _filtered_key_groups(other_df: pd.DataFrame, key_col: str, freq_cap: int) -> dict[str, list[str]]:
    """Precompute the exact-key channel's posting lists ONCE -- this depends
    only on the pool (`other_df`), never on S1, so it's identical for every
    S1 batch. Frequency-capped here too: a key whose posting list exceeds
    `freq_cap` will never be used by ANY batch, so drop it now rather than
    re-checking the length on every batch."""
    groups = other_df.groupby(key_col)["entity_id"].apply(list)
    return {key: ids for key, ids in groups.items() if key and len(ids) <= freq_cap}


def _exact_key_candidates_from_groups(
    s1_df: pd.DataFrame, other_groups: dict[str, list[str]], key_col: str
) -> dict[str, set[str]]:
    """Batched counterpart of `_exact_key_candidates`: takes the pool's
    already-grouped-and-capped posting lists instead of recomputing them."""
    result: dict[str, set[str]] = defaultdict(set)
    s1_by_key = s1_df.groupby(key_col)["entity_id"].apply(list)
    for key, s1_ids in s1_by_key.items():
        if not key:
            continue
        other_ids = other_groups.get(key)
        if not other_ids:
            continue
        for s1_id in s1_ids:
            result[s1_id].update(other_ids)
    return result


DEFAULT_POOL_MIN_NGRAM_DF = 2  # unlike MIN_NGRAM_DF=1 (tuned for small per-batch buckets),
# a pool-wide fit sees hundreds of thousands to millions of documents per country --
# requiring df>=1 there would retain an enormous number of single-occurrence n-grams
# (typos, OCR noise, rare foreign characters), bloating the cached vectorizer/matrix
# for no real recall benefit.


def _build_tfidf_pool_index(
    other_df: pd.DataFrame, text_col: str, min_df: int = DEFAULT_POOL_MIN_NGRAM_DF
) -> dict[str, dict]:
    """Fit + transform the TF-IDF matrix for the POOL side ONCE per country,
    cached for reuse by every S1 batch (`.transform()`, never re-fit).

    Deliberately fits on the pool's vocabulary only, not a union with S1's
    text (the non-batched `_tfidf_topk_candidates` fits on the union) --
    fitting on S1 too would require knowing all S1 batches upfront, defeating
    the point of caching. Any S1-side n-gram absent from the pool's
    vocabulary is simply dropped by `.transform()`, which mainly discards
    typo-only n-grams; the pool is far larger and more diverse than any
    single S1 batch, so this is a minor, well-understood approximation, not
    a blind one.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    index: dict[str, dict] = {}
    for country, group in other_df.groupby("country"):
        texts = group[text_col].tolist()
        if not any(texts):
            continue
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts).astype(np.float32)
        if matrix.nnz == 0:
            continue
        index[country] = {
            "vectorizer": vectorizer,
            "matrix": matrix.tocsr(),  # kept row-major (per-pool-row); transposed lazily at query time
            "entity_ids": group["entity_id"].to_numpy(),
        }
    return index


def _tfidf_topk_from_index(
    s1_df: pd.DataFrame,
    text_col: str,
    pool_index: dict[str, dict],
    top_k: int,
    threshold: float,
) -> dict[str, set[str]]:
    """Batched counterpart of `_tfidf_topk_candidates`: queries a cached,
    already-fitted pool-side TF-IDF index instead of fitting fresh."""
    from sparse_dot_topn import sp_matmul_topn

    result: dict[str, set[str]] = defaultdict(set)
    for country, s1_group in s1_df.groupby("country"):
        entry = pool_index.get(country)
        if entry is None or s1_group.empty:
            continue
        s1_texts = s1_group[text_col].tolist()
        if not any(s1_texts):
            continue
        s1_mat = entry["vectorizer"].transform(s1_texts).astype(np.float32)
        if s1_mat.nnz == 0:
            continue
        sims = sp_matmul_topn(
            s1_mat, entry["matrix"].T.tocsr(), top_n=top_k, threshold=threshold, n_threads=-1
        ).tocsr()
        s1_ids = s1_group["entity_id"].to_numpy()
        other_ids = entry["entity_ids"]
        for row_idx in range(sims.shape[0]):
            start, end = sims.indptr[row_idx], sims.indptr[row_idx + 1]
            if start == end:
                continue
            cols = sims.indices[start:end]
            result[s1_ids[row_idx]].update(other_ids[cols])
    return result


def build_pool_index(
    other_df: pd.DataFrame,
    freq_cap: int = DEFAULT_FREQ_CAP,
    min_df: int = DEFAULT_POOL_MIN_NGRAM_DF,
    use_tfidf: bool = True,
) -> PoolIndex:
    """Build the full cached index for one pool (S2 or S3): exact-key posting
    lists + 3 TF-IDF vectorizer/matrix pairs (name, name_skeleton, addr),
    ALL computed once from `other_df` alone. Pass the resulting `PoolIndex`
    to `generate_candidates_batched` for every S1 batch -- see that
    function's docstring and this module's `run_inference` caller for why
    this exists (repeatedly re-fitting TF-IDF on a multi-million-row pool
    once per S1 batch was the dominant cost of naive batching).

    `use_tfidf=False` skips fitting the 3 TF-IDF vectorizers entirely,
    leaving only the exact-key channels -- for a fast, deliberately
    lower-recall pass (e.g. a smoke-test submission to validate the output
    format / hidden-eval pipeline before spending the time for the full
    fuzzy-matching run): fitting TF-IDF over the entire multi-million-row
    pool is the dominant cost of building this index, independent of top_k
    or threshold, so this is the only way to meaningfully cut that cost.
    """
    progress.log(f"  pool index: name_key groups ({len(other_df)} rows)...")
    name_key_groups = _filtered_key_groups(other_df, "name_key", freq_cap)
    progress.log("  pool index: addr_key groups...")
    addr_key_groups = _filtered_key_groups(other_df, "addr_key", freq_cap)
    if use_tfidf:
        progress.log("  pool index: tfidf_name vectorizer...")
        tfidf_name = _build_tfidf_pool_index(other_df, "name_norm", min_df)
        progress.log("  pool index: tfidf_name_skeleton vectorizer...")
        tfidf_name_skeleton = _build_tfidf_pool_index(other_df, "name_skeleton", min_df)
        progress.log("  pool index: tfidf_addr vectorizer...")
        tfidf_addr = _build_tfidf_pool_index(other_df, "addr_norm", min_df)
    else:
        progress.log("  pool index: skipping TF-IDF vectorizers (use_tfidf=False, exact-key-only fast mode)")
        tfidf_name = tfidf_name_skeleton = tfidf_addr = {}
    return PoolIndex(name_key_groups, addr_key_groups, tfidf_name, tfidf_name_skeleton, tfidf_addr)


def generate_candidates_batched(
    s1_batch_df: pd.DataFrame,
    pool_index: PoolIndex,
    top_k: int = DEFAULT_TOP_K,
    tfidf_threshold: float = 0.1,
) -> dict[str, set[str]]:
    """Batched counterpart of `generate_candidates_for_source`: same 5
    channels, unioned the same way, but querying a precomputed `PoolIndex`
    instead of recomputing pool-side structures for every S1 batch."""
    channels = {
        "name_key": _exact_key_candidates_from_groups(s1_batch_df, pool_index.name_key_groups, "name_key"),
        "addr_key": _exact_key_candidates_from_groups(s1_batch_df, pool_index.addr_key_groups, "addr_key"),
        "tfidf_name": _tfidf_topk_from_index(s1_batch_df, "name_norm", pool_index.tfidf_name, top_k, tfidf_threshold),
        "tfidf_name_skeleton": _tfidf_topk_from_index(
            s1_batch_df, "name_skeleton", pool_index.tfidf_name_skeleton, top_k, tfidf_threshold
        ),
        "tfidf_addr": _tfidf_topk_from_index(s1_batch_df, "addr_norm", pool_index.tfidf_addr, top_k, tfidf_threshold),
    }
    merged: dict[str, set[str]] = defaultdict(set)
    for channel_result in channels.values():
        for s1_id, cand_ids in channel_result.items():
            merged[s1_id].update(cand_ids)
    return merged


def generate_all_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    freq_cap: int = DEFAULT_FREQ_CAP,
    tfidf_threshold: float = 0.1,
    n_jobs: int = DEFAULT_N_JOBS,
) -> dict[str, set[str]]:
    """Full candidate_pairs structure: source1_entity_id -> {S2/S3 candidate ids}.

    Uses the lightweight `add_blocking_representations` (blocking never needs
    the full per-record representation -- see that function's docstring).
    """
    with parallel_pool(n_jobs) as pool:
        s1_r = add_blocking_representations(s1_df, n_jobs, pool=pool)
        s2_r = add_blocking_representations(s2_df, n_jobs, pool=pool)
        s3_r = add_blocking_representations(s3_df, n_jobs, pool=pool)

    progress.log("Blocking S1 vs S2...")
    cand_s2 = generate_candidates_for_source(s1_r, s2_r, top_k, freq_cap, tfidf_threshold)
    progress.log("Blocking S1 vs S3...")
    cand_s3 = generate_candidates_for_source(s1_r, s3_r, top_k, freq_cap, tfidf_threshold)

    merged: dict[str, set[str]] = defaultdict(set)
    for s1_id in s1_r["entity_id"]:
        merged[s1_id] = cand_s2.get(s1_id, set()) | cand_s3.get(s1_id, set())
    return dict(merged)
