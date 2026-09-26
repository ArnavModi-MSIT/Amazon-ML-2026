"""Data-driven blocking design: over the TRUE pairs of 20k held-out train S1
entities vs the FULL train pools, measure how much recall each blocking
signal gives and how many candidates it would generate.

Found necessary because real-scale validation showed a 0.33 recall ceiling
for the current Splink blocking: absolute frequency caps (token df<=50, key
count<=30) tuned at 750k-pool scale mask most meaningful tokens at 5M scale.

Signals analysed (all within same country -- 0 cross-country true pairs):
  - exact name_key / addr_key
  - shared name token (from name_norm AND name_skeleton, so cross-script
    pairs can share tokens) at various absolute df caps
  - shared addr token at various df caps
  - adaptive: pair shares one of the S1 record's k RAREST tokens (no
    absolute cap -- scale-free), for k in {1,2,3}
Candidate volume for token channels is the upper bound sum(df) over the
tokens used per S1 entity.

Run: python -u -m src._blocking_recall_analysis
"""
from collections import Counter

import numpy as np

from . import blocking, config, io_utils, progress
from .parallel_utils import DEFAULT_N_JOBS, parallel_pool

N_VAL = 20_000
MIN_LEN = 3
CAPS = [50, 200, 1000, 5000, 20000, 10**12]


def _tokens(norm: str, skel: str) -> set[str]:
    return {t for t in (norm.split() + skel.split()) if len(t) >= MIN_LEN}


def _addr_tokens(norm: str) -> set[str]:
    return {t for t in norm.split() if len(t) >= MIN_LEN}


def main() -> None:
    rng = np.random.default_rng(config.RANDOM_SEED)
    progress.log("Loading...")
    s1_all = io_utils.load_source(config.TRAIN_SOURCE1)
    pools = {"S2": io_utils.load_source(config.TRAIN_SOURCE2), "S3": io_utils.load_source(config.TRAIN_SOURCE3)}
    all_true = io_utils.build_true_matches(io_utils.load_ground_truth())

    ck_idx = rng.choice(len(s1_all), size=50_000, replace=False)
    excluded = set(s1_all.iloc[ck_idx]["entity_id"])
    rest = s1_all[~s1_all["entity_id"].isin(excluded)]
    val = rest.iloc[np.random.default_rng(123).choice(len(rest), size=N_VAL, replace=False)].reset_index(drop=True)
    del s1_all, rest

    with parallel_pool(DEFAULT_N_JOBS) as pool:
        vr = blocking.add_blocking_representations(val, DEFAULT_N_JOBS, pool=pool)
        prep = {k: blocking.add_blocking_representations(v, DEFAULT_N_JOBS, pool=pool) for k, v in pools.items()}
    del pools

    s1_name = {e: _tokens(n, s) for e, n, s in zip(vr["entity_id"], vr["name_norm"], vr["name_skeleton"])}
    s1_addr = {e: _addr_tokens(a) for e, a in zip(vr["entity_id"], vr["addr_norm"])}
    s1_nk = dict(zip(vr["entity_id"], vr["name_key"]))
    s1_ak = dict(zip(vr["entity_id"], vr["addr_key"]))

    for src, rep in prep.items():
        progress.log(f"===== {src} =====")
        # document frequency per (country, token) over the full pool
        ndf, adf = Counter(), Counter()
        nk_cnt = Counter(zip(rep["country"], rep["name_key"]))
        ak_cnt = Counter(zip(rep["country"], rep["addr_key"]))
        for c, n, s, a in zip(rep["country"], rep["name_norm"], rep["name_skeleton"], rep["addr_norm"]):
            for t in _tokens(n, s):
                ndf[(c, t)] += 1
            for t in _addr_tokens(a):
                adf[(c, t)] += 1
        progress.log(f"  df built: {len(ndf)} name tokens, {len(adf)} addr tokens")

        by_id = rep.set_index("entity_id")
        country_of = dict(zip(vr["entity_id"], vr["country"]))
        pairs = [(e, m) for e in vr["entity_id"] for m in all_true.get(e, ()) if m.startswith(src + "-")]
        m_rows = by_id.loc[[m for _, m in pairs], ["name_norm", "name_skeleton", "addr_norm", "name_key", "addr_key"]]
        n_pairs = len(pairs)

        min_name_df, min_addr_df, nk_hit, ak_hit = [], [], [], []
        rarest_rank = []  # rank (0-based) of the rarest shared name token among s1's tokens sorted by df
        for (e, m), row in zip(pairs, m_rows.itertuples(index=False)):
            c = country_of[e]
            shared_n = s1_name[e] & _tokens(row.name_norm, row.name_skeleton)
            shared_a = s1_addr[e] & _addr_tokens(row.addr_norm)
            min_name_df.append(min((ndf[(c, t)] for t in shared_n), default=np.inf))
            min_addr_df.append(min((adf[(c, t)] for t in shared_a), default=np.inf))
            nk_hit.append(bool(s1_nk[e]) and s1_nk[e] == row.name_key)
            ak_hit.append(bool(s1_ak[e]) and s1_ak[e] == row.addr_key)
            order = sorted(s1_name[e], key=lambda t: ndf[(c, t)])
            ranks = [order.index(t) for t in shared_n]
            rarest_rank.append(min(ranks) if ranks else np.inf)
        min_name_df, min_addr_df = np.array(min_name_df), np.array(min_addr_df)
        nk_hit, ak_hit, rarest_rank = np.array(nk_hit), np.array(ak_hit), np.array(rarest_rank)

        progress.log(f"  true pairs: {n_pairs}")
        progress.log(f"  exact name_key: {nk_hit.mean():.3f}   exact addr_key: {ak_hit.mean():.3f}   "
                     f"either: {(nk_hit | ak_hit).mean():.3f}")
        progress.log(f"  share ANY name token (no cap): {np.isfinite(min_name_df).mean():.3f}   "
                     f"share ANY addr token: {np.isfinite(min_addr_df).mean():.3f}   "
                     f"name OR addr token: {(np.isfinite(min_name_df) | np.isfinite(min_addr_df)).mean():.3f}")

        # candidate volume upper bounds per S1 entity for each cap
        vol_n = {cap: [] for cap in CAPS}
        vol_a = {cap: [] for cap in CAPS}
        vol_k = {k: [] for k in (1, 2, 3)}
        for e in vr["entity_id"]:
            c = country_of[e]
            dfs_n = sorted(ndf[(c, t)] for t in s1_name[e])
            dfs_a = [adf[(c, t)] for t in s1_addr[e]]
            for cap in CAPS:
                vol_n[cap].append(sum(d for d in dfs_n if d <= cap))
                vol_a[cap].append(sum(d for d in dfs_a if d <= cap))
            for k in (1, 2, 3):
                vol_k[k].append(sum(dfs_n[:k]))

        progress.log("  cap      | name-tok recall | addr-tok recall | name|addr|keys recall | avg cand/entity (name+addr, upper bound)")
        for cap in CAPS:
            rn = min_name_df <= cap
            ra = min_addr_df <= cap
            union = rn | ra | nk_hit | ak_hit
            progress.log(f"  {cap:>8} | {rn.mean():15.3f} | {ra.mean():15.3f} | {union.mean():22.3f} | "
                         f"{np.mean(vol_n[cap]) + np.mean(vol_a[cap]):12.0f}  (median {np.median(np.array(vol_n[cap]) + np.array(vol_a[cap])):.0f})")
        progress.log("  adaptive k-rarest name tokens (no absolute cap):")
        for k in (1, 2, 3):
            rk = rarest_rank < k
            progress.log(f"    k={k}: name recall {rk.mean():.3f} | with keys {(rk | nk_hit | ak_hit).mean():.3f} | "
                         f"avg cand/entity {np.mean(vol_k[k]):.0f} (median {np.median(vol_k[k]):.0f})")
        # how dissimilar are the unreachable pairs?
        unreach = ~(np.isfinite(min_name_df) | np.isfinite(min_addr_df) | nk_hit | ak_hit)
        progress.log(f"  pairs sharing NO token and no key: {unreach.mean():.3f}")
        if unreach.any():
            idx = np.where(unreach)[0][:8]
            for i in idx:
                e, m = pairs[i]
                progress.log(f"    S1 {vr.loc[vr['entity_id']==e, 'name_norm'].iloc[0]!r} | "
                             f"{vr.loc[vr['entity_id']==e, 'addr_norm'].iloc[0]!r}")
                progress.log(f"    {src} {m_rows.iloc[i]['name_norm']!r} | {m_rows.iloc[i]['addr_norm']!r}")
    progress.log("DONE")


if __name__ == "__main__":
    main()
