"""Blocking experiment: IDF-weighted word-token cosine, top-K per S1 entity.

Recall analysis at real pool scale showed ~100% of true pairs share at
least one word token (name or address), but the shared tokens are often
common -- absolute df caps trade recall for enormous candidate volume.
Ranking pool records by TF-IDF cosine over WORD tokens and keeping the
top-K per S1 entity lets rare shared tokens dominate while bounding
candidates/entity directly (K), which the final ranking now rewards.

Per country: fit one word-token TF-IDF on the pool, drop ultra-common
tokens post-hoc (several df thresholds, one fit), top-50 via
sparse_dot_topn, report recall@K / candidates-per-entity / timings.

Run: python -u -m src._idf_topk_blocking_eval
"""
import time

import numpy as np
from scipy import sparse

from . import blocking, config, io_utils, progress
from .parallel_utils import DEFAULT_N_JOBS, parallel_pool

N_VAL = 20_000
TOP_N = 50
KS = [5, 10, 20, 50]
MAX_DFS = [2_000, 10_000, 50_000, 10**12]


def _doc(name_norm: str, skel: str, addr: str) -> list[str]:
    toks = {"n:" + t for t in name_norm.split() if len(t) >= 2}
    toks |= {"n:" + t for t in skel.split() if len(t) >= 2}
    toks |= {"a:" + t for t in addr.split() if len(t) >= 2}
    return list(toks)


def _mask_and_norm(mat: sparse.csr_matrix, keep: np.ndarray) -> sparse.csr_matrix:
    m = mat @ sparse.diags(keep.astype(np.float32))
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.diags((1.0 / norms).astype(np.float32)) @ m


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn

    rng = np.random.default_rng(config.RANDOM_SEED)
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

    for src, rep in prep.items():
        progress.log(f"===== {src} =====")
        n_true = 0
        # rank_of[(s1, cand)] -> best rank under each max_df setting
        ranks = {mdf: {} for mdf in MAX_DFS}
        nnz_rows = {mdf: [] for mdf in MAX_DFS}
        t_fit = t_mm = 0.0
        for country in sorted(set(vr["country"])):
            v_c = vr[vr["country"] == country]
            p_c = rep[rep["country"] == country]
            if v_c.empty or p_c.empty:
                continue
            t0 = time.time()
            vec = TfidfVectorizer(analyzer=lambda d: d, sublinear_tf=True, dtype=np.float32)
            pool_docs = [_doc(a, b, c) for a, b, c in zip(p_c["name_norm"], p_c["name_skeleton"], p_c["addr_norm"])]
            P = vec.fit_transform(pool_docs).tocsr()
            del pool_docs
            Q = vec.transform([_doc(a, b, c) for a, b, c in zip(v_c["name_norm"], v_c["name_skeleton"], v_c["addr_norm"])]).tocsr()
            df = np.bincount(P.indices, minlength=P.shape[1])
            t_fit += time.time() - t0
            p_ids = p_c["entity_id"].to_numpy()
            q_ids = v_c["entity_id"].to_numpy()
            progress.log(f"  {country}: pool {P.shape[0]} x vocab {P.shape[1]}, nnz {P.nnz}; fit {time.time()-t0:.0f}s")
            for mdf in MAX_DFS:
                keep = df <= mdf
                Pm, Qm = _mask_and_norm(P, keep), _mask_and_norm(Q, keep)
                t1 = time.time()
                S = sp_matmul_topn(Qm, Pm.T.tocsr(), top_n=TOP_N, threshold=0.0, n_threads=-1).tocsr()
                t_mm += time.time() - t1
                progress.log(f"    max_df={mdf}: kept {keep.sum()} tokens, matmul {time.time()-t1:.1f}s")
                for r in range(S.shape[0]):
                    s, e = S.indptr[r], S.indptr[r + 1]
                    nnz_rows[mdf].append(e - s)
                    if s == e:
                        continue
                    order = np.argsort(-S.data[s:e])
                    cols = S.indices[s:e][order]
                    qid = q_ids[r]
                    for rank, col in enumerate(cols):
                        ranks[mdf][(qid, p_ids[col])] = rank
        true_pairs = [(e, m) for e in vr["entity_id"] for m in all_true.get(e, ()) if m.startswith(src + "-")]
        n_true = len(true_pairs)
        progress.log(f"  true pairs {n_true}; total fit {t_fit:.0f}s, total matmul {t_mm:.0f}s (for {N_VAL} S1 rows)")
        progress.log("  max_df     | " + " | ".join(f"R@{k:<3} cand/ent" for k in KS))
        for mdf in MAX_DFS:
            rk = np.array([ranks[mdf].get(p, np.inf) for p in true_pairs])
            nr = np.array(nnz_rows[mdf])
            cells = [f"{(rk < k).mean():.3f} {np.minimum(nr, k).mean():5.1f}" for k in KS]
            progress.log(f"  {mdf:>10} | " + " | ".join(cells))
    progress.log("DONE")


if __name__ == "__main__":
    main()
