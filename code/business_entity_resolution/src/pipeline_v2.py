"""v2-v4 pipeline: IDF top-K blocking -> (optional) relative-floor pruning ->
pairwise features + blocking, pool-frequency and within-entity features ->
LightGBM -> single threshold -> global one-to-one conflict resolution.

Design decisions, each from a real-scale measurement:
  - Blocking: blocking_idf (recall ceiling 0.33 -> ~0.95 at K=10/source;
    0.940 after pruning in v4, up from 0.924 in v3).
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
from . import text_repr as tr
from .parallel_utils import DEFAULT_N_JOBS, parallel_map

BLOCK_FEATURES = ["block_score", "block_rank", "block_top_score", "block_score_gap"]
FREQ_FEATURES = ["log_cand_name_key_freq", "log_cand_addr_key_freq", "log_s1_name_key_pool_freq",
                 "name_freq_x_addr_missing"]
LIST_FEATURES = ["block_ratio", "block_gap_top2", "n_within_90", "name_sort_rel", "name_lev_rel", "addr_tok_rel"]
# How many S1 businesses (whole S1 file of the split) share a name / a house-number+street: a name unique
# in S1 makes an empty-address pool record with that name almost surely its match; one S1 business at an
# address makes a differently named pool record there likely a renamed duplicate, several = co-located.
# Plus: how many OTHER S1 businesses carry the candidate's exact name (it is theirs), and the share of the
# candidate's name words that exist anywhere in S1 (invented replacement names like "Zephavi" have none).
# Counts are clipped: test pools are 0.62x train for the US, so raw counts would shift at test time.
UNIVERSE_FEATURES = ["log_cand_core_s1_freq", "log_cand_core_pool_freq", "log_s1_core_s1_freq",
                     "log_cand_addrkey_s1_freq", "log_s1_addrkey_s1_freq", "log_cand_core_other_s1",
                     "cand_tok_s1_frac", "log_cand_addrfull_s1_freq", "log_cand_addrfull_pool_freq",
                     "log_cand_core_pool_naddr", "log_cand_addr_pool_nnames"]
# (last two: distinct house-number|street keys per name in the pool -- a chain vs one business -- and
# distinct names per house-number|street -- one business vs a shared building)
POOL_COUNT_CLIP, S1_COUNT_CLIP = 8, 4
# key_channel: pair found only by the exact-key pass (no cosine score); g_same_*: how many of the entity's
# other candidates share this candidate's name core / full address (duplicates of one business cluster)
KEY_FEATURES = ["key_channel", "g_same_name", "g_same_addr", "empty_channel"]
# Reverse pass: each pool record's single best S1 entity among the WHOLE S1 file (same cosine). rev_top1 = this
# entity is it; rev_other = another S1 entity is it (evidence against); rev_best_ratio = this pair's cosine
# relative to the record's best. Measured: +0.39 candidates/entity took the blocking recall ceiling 0.965 -> 0.978.
REVERSE_FEATURES = ["rev_top1", "rev_other", "rev_best_ratio"]
FEATURE_NAMES_ALL = (list(features.FEATURE_NAMES) + BLOCK_FEATURES + FREQ_FEATURES + LIST_FEATURES
                     + UNIVERSE_FEATURES + KEY_FEATURES + REVERSE_FEATURES)
KEY_CAP = 5  # an exact name+number key shared by more pool records than this is not specific


def _name_num_keys(rep: pd.DataFrame) -> list[list[str]]:
    """Exact keys: name core + house number, sorted name core + house number, name core alone."""
    out = []
    for cty, nn, an in zip(rep["country"].astype(str), rep["name_norm"], rep["addr_norm"]):
        toks = [t for t in nn.split() if t not in tr.LEGAL_SUFFIXES]
        core, srt = "".join(toks), "".join(sorted(toks))
        num = tr.address_number_key(an)[0]
        ks = []
        if len(core) >= 4 and num:
            ks += [f"{cty}|cn|{core}|{num}", f"{cty}|sn|{srt}|{num}"]
        if len(core) >= 6:
            ks.append(f"{cty}|c|{core}")
        out.append(ks)
    return out


def _key_pairs(s1_rep: pd.DataFrame, s1_keys: list[list[str]], rep: pd.DataFrame) -> pd.DataFrame:
    """(S1, pool) pairs sharing a specific exact key (idea from public competition repos: exact-key passes
    next to TF-IDF catch pairs whose cosine rank is crowded out). Name+number keys: <= KEY_CAP pool
    records; name-only key: only when the name is unique in the pool."""
    wanted = {k for ks in s1_keys for k in ks}
    idx: dict[str, list] = {}
    for eid, ks in zip(rep["entity_id"], _name_num_keys(rep)):
        for k in ks:
            if k in wanted:
                idx.setdefault(k, []).append(eid)
    a, b = [], []
    for e, ks in zip(s1_rep["entity_id"], s1_keys):
        seen = set()
        for k in ks:
            ids = idx.get(k, ())
            if 0 < len(ids) <= (1 if "|c|" in k else KEY_CAP):
                seen.update(ids)
        a += [e] * len(seen)
        b += list(seen)
    return pd.DataFrame({"source1_entity_id": a, "candidate_entity_id": b})


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


def _core_and_addr_keys(rep: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """country-prefixed name-core, house-number|street, and full-address (sorted token set, >= 3 tokens)
    keys ("" if missing)."""
    cty = rep["country"].astype(str).to_numpy()
    core = [c + "|" + k if k else "" for c, k in zip(cty, map(tr.name_core, rep["name_norm"]))]
    addr, full = [], []
    for c, a in zip(cty, rep["addr_norm"]):
        num, street = tr.address_number_key(a)
        addr.append(f"{c}|{num}|{street}" if num and street else "")
        toks = sorted(set(a.split()))
        full.append(c + "|" + " ".join(toks) if len(toks) >= 3 else "")
    idx = rep.index
    return pd.Series(core, index=idx), pd.Series(addr, index=idx), pd.Series(full, index=idx)


def _tok_s1_frac(rep: pd.DataFrame, s1_tokens: set) -> pd.Series:
    """Share of a record's ASCII name words (>= 3 chars, not legal suffixes) found in the S1 word set of its
    country; -1 if it has none."""
    out = []
    for c, name in zip(rep["country"].astype(str), rep["name_norm"]):
        toks = [t for t in name.split() if len(t) >= 3 and t.isascii() and t not in tr.LEGAL_SUFFIXES]
        out.append(sum(f"{c}|{t}" in s1_tokens for t in toks) / len(toks) if toks else -1.0)
    return pd.Series(np.asarray(out, dtype=np.float32), index=rep["entity_id"].to_numpy())


# country -> {"name": {token: idf}, "addr": {token: idf}, "n": pool size}; filled by generate_candidates from the
# pools of the current split and read by featurize (same process) to attach per-record token weights.
_IDF: dict[str, dict] = {}


def _accumulate_df(rep: pd.DataFrame, df_counts: dict) -> None:
    for cty, nn, an in zip(rep["country"].astype(str), rep["name_norm"], rep["addr_norm"]):
        d = df_counts.setdefault(cty, {"name": {}, "addr": {}, "n": 0})
        d["n"] += 1
        for t in {t for t in nn.split() if t not in tr.LEGAL_SUFFIXES}:
            d["name"][t] = d["name"].get(t, 0) + 1
        for t in set(an.split()):
            d["addr"][t] = d["addr"].get(t, 0) + 1


def _finalize_idf(df_counts: dict) -> None:
    _IDF.clear()
    for cty, d in df_counts.items():
        n = d["n"]
        _IDF[cty] = {"n": n, **{f: {t: float(np.log(n / c)) for t, c in d[f].items()} for f in ("name", "addr")}}


def _attach_idf(lookup: dict, name_field: str) -> None:
    """Add name_idf / addr_idf {token: idf} to each record dict (unseen tokens get the maximum idf)."""
    for rec in lookup.values():
        d = _IDF.get(str(rec.get("country", "")))
        if d is None:
            rec["name_idf"], rec["addr_idf"] = {}, {}
            continue
        mx = float(np.log(max(d["n"], 2)))
        rec["name_idf"] = {t: d["name"].get(t, mx) for t in rec[name_field]}
        rec["addr_idf"] = {t: d["addr"].get(t, mx) for t in set(rec["addr_norm"].split())}


def _count_map(keys: pd.Series, lookup: pd.Series, ids) -> pd.Series:
    """For each key in `lookup`, its count in `keys` (0 for missing keys), indexed by `ids`."""
    counts = keys[keys != ""].value_counts()
    return pd.Series(lookup.map(counts).fillna(0).astype(np.float32).to_numpy(), index=ids)


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    k: int = blocking_idf.DEFAULT_K,
    max_df: int = blocking_idf.DEFAULT_MAX_DF,
    n_jobs: int = DEFAULT_N_JOBS,
    pool=None,
    s1_universe: pd.DataFrame | None = None,
    field_weights: tuple[float, ...] | None = None,
    segment: bool = False,
    latin_only_tokens: bool = False,
    key_channel: bool = False,
    reverse: bool = False,
    empty_k: int = 0,
) -> pd.DataFrame:
    """All candidate pairs (both pools) with blocking scores and key-frequency
    columns. `s1_universe` = the whole S1 file of the split (default: s1_df,
    which at inference IS the whole test S1); S1-frequency features are
    counted over it so training (a subset of train S1) matches inference."""
    progress.log(f"[v2] blocking reps: S1 {len(s1_df)}, S2 {len(s2_df)}, S3 {len(s3_df)}")
    s1_rep = blocking.add_blocking_representations(s1_df, n_jobs, pool=pool)
    uni_rep = s1_rep if s1_universe is None else blocking.add_blocking_representations(s1_universe, n_jobs, pool=pool)
    uni_core, uni_addr, uni_full = _core_and_addr_keys(uni_rep)
    s1_core, s1_addr, _ = _core_and_addr_keys(s1_rep)
    s1_ids = s1_rep["entity_id"].to_numpy()
    s1_core_freq = _count_map(uni_core, s1_core, s1_ids)
    s1_addr_freq = _count_map(uni_addr, s1_addr, s1_ids)
    s1_tokens = {f"{c}|{t}" for c, name in zip(uni_rep["country"].astype(str), uni_rep["name_norm"])
                 for t in name.split() if len(t) >= 3}
    seg_costs = blocking_idf.build_segment_costs(uni_rep) if segment else None
    s1_keys = _name_num_keys(s1_rep) if key_channel else None
    rev_rep = uni_rep if reverse else None  # reverse ranks each pool record against the WHOLE S1 file
    s1_id_set = set(s1_ids)
    del uni_rep, s1_core, s1_addr
    parts = []
    df_counts: dict = {}
    for label, other in (("S2", s2_df), ("S3", s3_df)):
        rep = blocking.add_blocking_representations(other, n_jobs, pool=pool)
        _accumulate_df(rep, df_counts)
        if seg_costs is not None:
            rep["name_seg"] = blocking_idf.segment_names(rep, seg_costs)
            progress.log(f"[v2] segmented {int((rep['name_seg'] != '').sum())} glued {label} names")
        progress.log(f"[v2] IDF top-{k} vs {label} (max_df={max_df})...")
        c = blocking_idf.topk_candidates(s1_rep, rep, k=k, max_df=max_df, label=f"-{label}",
                                         field_weights=field_weights, latin_only_tokens=latin_only_tokens,
                                         empty_k=empty_k)
        c["key_channel"] = np.float32(0)
        if s1_keys is not None:
            kp = _key_pairs(s1_rep, s1_keys, rep)
            known = set(zip(c["source1_entity_id"], c["candidate_entity_id"]))
            kp = kp[[p not in known for p in zip(kp["source1_entity_id"], kp["candidate_entity_id"])]]
            top = c.groupby("source1_entity_id")["block_top_score"].first()
            kp = kp.assign(block_score=np.float32(0), block_rank=np.int16(k),
                           block_top_score=kp["source1_entity_id"].map(top).fillna(0).astype(np.float32),
                           key_channel=np.float32(1))
            progress.log(f"[v2] exact-key channel vs {label}: +{len(kp)} pairs")
            c = pd.concat([c, kp], ignore_index=True)
        if rev_rep is not None:
            r = blocking_idf.reverse_topk(rev_rep, rep, k=1, label=f"-{label}", field_weights=field_weights,
                                          latin_only_tokens=latin_only_tokens)
            win = pd.Series(r["source1_entity_id"].to_numpy(), index=r["candidate_entity_id"].to_numpy())
            wsc = pd.Series(r["rev_score"].to_numpy(), index=r["candidate_entity_id"].to_numpy())
            covered = set(c["candidate_entity_id"].to_numpy()[
                win.reindex(c["candidate_entity_id"]).to_numpy() == c["source1_entity_id"].to_numpy()])
            new = r[r["source1_entity_id"].isin(s1_id_set) & ~r["candidate_entity_id"].isin(covered)]
            top = c.groupby("source1_entity_id")["block_top_score"].first()
            new = pd.DataFrame({
                "source1_entity_id": new["source1_entity_id"].to_numpy(),
                "candidate_entity_id": new["candidate_entity_id"].to_numpy(),
                "block_score": new["rev_score"].to_numpy(np.float32),  # the reverse cosine IS this pair's cosine
                "block_rank": np.int16(k),
                "block_top_score": new["source1_entity_id"].map(top).fillna(0).to_numpy(np.float32),
                "key_channel": np.float32(0),
            })
            progress.log(f"[v2] reverse channel vs {label}: +{len(new)} pairs")
            c = pd.concat([c, new], ignore_index=True)
            w = win.reindex(c["candidate_entity_id"]).to_numpy()
            s1v = c["source1_entity_id"].to_numpy()
            c["rev_top1"] = (w == s1v).astype(np.float32)
            c["rev_other"] = (pd.notna(w) & (w != s1v)).astype(np.float32)
            c["rev_best_score"] = wsc.reindex(c["candidate_entity_id"]).fillna(0).to_numpy(np.float32)
            del r, win, wsc, covered, new, w, s1v
        pool_nfreq, s1_nfreq = _key_freqs(rep, "name_key", s1_rep)
        pool_afreq = _key_freqs(rep, "addr_key")
        c["cand_name_key_freq"] = pool_nfreq.reindex(c["candidate_entity_id"]).to_numpy()
        c["cand_addr_key_freq"] = pool_afreq.reindex(c["candidate_entity_id"]).to_numpy()
        c["s1_name_key_pool_freq"] = s1_nfreq.reindex(c["source1_entity_id"]).to_numpy()
        p_core, p_addr, p_full = _core_and_addr_keys(rep)
        p_ids = rep["entity_id"].to_numpy()
        c["cand_core_s1_freq"] = _count_map(uni_core, p_core, p_ids).reindex(c["candidate_entity_id"]).to_numpy()
        c["cand_core_pool_freq"] = _count_map(p_core, p_core, p_ids).reindex(c["candidate_entity_id"]).to_numpy()
        c["cand_addrkey_s1_freq"] = _count_map(uni_addr, p_addr, p_ids).reindex(c["candidate_entity_id"]).to_numpy()
        c["s1_core_s1_freq"] = s1_core_freq.reindex(c["source1_entity_id"]).to_numpy()
        c["s1_addrkey_s1_freq"] = s1_addr_freq.reindex(c["source1_entity_id"]).to_numpy()
        c["cand_tok_s1_frac"] = _tok_s1_frac(rep, s1_tokens).reindex(c["candidate_entity_id"]).to_numpy()
        c["cand_addrfull_s1_freq"] = _count_map(uni_full, p_full, p_ids).reindex(c["candidate_entity_id"]).to_numpy()
        c["cand_addrfull_pool_freq"] = _count_map(p_full, p_full, p_ids).reindex(c["candidate_entity_id"]).to_numpy()
        both = pd.DataFrame({"core": p_core.to_numpy(), "addr": p_addr.to_numpy()})
        both = both[(both["core"] != "") & (both["addr"] != "")]
        naddr = both.groupby("core")["addr"].nunique()
        nnames = both.groupby("addr")["core"].nunique()
        c["cand_core_pool_naddr"] = pd.Series(p_core.map(naddr).fillna(0).to_numpy(np.float32), index=p_ids) \
            .reindex(c["candidate_entity_id"]).to_numpy()
        c["cand_addr_pool_nnames"] = pd.Series(p_addr.map(nnames).fillna(0).to_numpy(np.float32), index=p_ids) \
            .reindex(c["candidate_entity_id"]).to_numpy()
        del both, naddr, nnames
        del p_core, p_addr, p_full
        c["source"] = label
        parts.append(c)
        del rep, pool_nfreq, s1_nfreq, pool_afreq
    _finalize_idf(df_counts)
    del df_counts
    cands = pd.concat(parts, ignore_index=True)
    if "empty_channel" in cands:  # exact-key / reverse rows have no flag
        cands["empty_channel"] = cands["empty_channel"].fillna(0).astype(np.float32)
    progress.log(f"[v2] candidates: {len(cands)} pairs ({len(cands) / max(len(s1_df), 1):.2f}/entity)")
    return cands


def prune_candidates(cands: pd.DataFrame, rel_floor, min_rank: int, country_of: pd.Series | None = None) -> pd.DataFrame:
    """Keep ranks < min_rank, plus any candidate with block_score >=
    rel_floor * block_top_score (block_top_score is per entity per pool).
    `rel_floor` may be a {country: floor, "default": floor} dict (needs
    `country_of`: S1 entity_id -> country): India's field-weighted scores are
    flatter, so it needs a higher floor to keep ~the same candidate count."""
    if rel_floor is None:
        return cands
    if isinstance(rel_floor, dict):
        floor = (cands["source1_entity_id"].map(country_of).map(rel_floor)
                 .fillna(rel_floor.get("default", 0.5)).to_numpy(np.float32))
    else:
        floor = rel_floor
    keep = (cands["block_rank"] < min_rank) | (cands["block_score"] >= floor * cands["block_top_score"])
    if "key_channel" in cands:
        keep |= cands["key_channel"].to_numpy() == 1  # exact-key pairs have no cosine score; always kept
    if "rev_top1" in cands:
        keep |= cands["rev_top1"].to_numpy() == 1  # the pool record's own best S1 entity; always kept
    if "empty_channel" in cands:
        keep |= cands["empty_channel"].to_numpy() == 1  # best empty-address pool records; always kept
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
    _attach_idf(s1_lookup, "name_tokens_no_suffix")
    _attach_idf(other_lookup, "name_tokens_no_suffix")
    cand_core = [tr.name_core(other_lookup[b]["name_norm"]) for b in cands["candidate_entity_id"]]
    cand_addr = [other_lookup[b]["addr_norm"] for b in cands["candidate_entity_id"]]
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

    def clog(col, clip):
        return np.log1p(np.minimum(cands[col].to_numpy(np.float32), clip))

    X["log_cand_name_key_freq"] = clog("cand_name_key_freq", POOL_COUNT_CLIP)
    X["log_cand_addr_key_freq"] = clog("cand_addr_key_freq", POOL_COUNT_CLIP)
    X["log_s1_name_key_pool_freq"] = clog("s1_name_key_pool_freq", POOL_COUNT_CLIP)
    X["name_freq_x_addr_missing"] = X["log_cand_name_key_freq"] * X["addr_missing_either"]
    if "cand_core_s1_freq" in cands:  # absent only in candidate tables cached before these features existed
        for col, clip in (("cand_core_s1_freq", S1_COUNT_CLIP), ("cand_core_pool_freq", POOL_COUNT_CLIP),
                          ("s1_core_s1_freq", S1_COUNT_CLIP), ("cand_addrkey_s1_freq", S1_COUNT_CLIP),
                          ("s1_addrkey_s1_freq", S1_COUNT_CLIP), ("cand_addrfull_s1_freq", S1_COUNT_CLIP),
                          ("cand_addrfull_pool_freq", POOL_COUNT_CLIP), ("cand_core_pool_naddr", POOL_COUNT_CLIP),
                          ("cand_addr_pool_nnames", POOL_COUNT_CLIP)):
            X["log_" + col] = clog(col, clip)
        other = np.maximum(cands["cand_core_s1_freq"].to_numpy(np.float32) - X["name_core_match"].to_numpy(), 0)
        X["log_cand_core_other_s1"] = np.log1p(np.minimum(other, S1_COUNT_CLIP))
        X["cand_tok_s1_frac"] = cands["cand_tok_s1_frac"].to_numpy(np.float32)

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
    X["key_channel"] = cands["key_channel"].to_numpy(np.float32) if "key_channel" in cands else np.float32(0)
    X["empty_channel"] = cands["empty_channel"].to_numpy(np.float32) if "empty_channel" in cands else np.float32(0)
    if "rev_top1" in cands:
        X["rev_top1"] = cands["rev_top1"].to_numpy(np.float32)
        X["rev_other"] = cands["rev_other"].to_numpy(np.float32)
        best = cands["rev_best_score"].to_numpy(np.float32)
        X["rev_best_ratio"] = np.where(best > 0, np.minimum(X["block_score"].to_numpy() / np.maximum(best, 1e-6), 2),
                                       np.float32(-1))
    else:
        X["rev_top1"] = X["rev_other"] = X["rev_best_ratio"] = np.float32(0)
    ent = cands["source1_entity_id"].to_numpy()
    for col, vals in (("g_same_name", cand_core), ("g_same_addr", cand_addr)):
        v = pd.Series(vals)
        cnt = v.groupby([ent, v.to_numpy()], sort=False).transform("size").to_numpy(np.float32) - 1
        X[col] = np.where(v.to_numpy() != "", cnt, np.float32(-1))
    return X.astype(np.float32)
