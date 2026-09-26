"""Cache the v2 validation set once: the fixed 20k held-out train S1 entities
blocked (K=10) against the FULL train pools, scored by artifacts/model_v2.txt.

Writes reports/val_v2.parquet with one row per candidate pair:
source1_entity_id, candidate_entity_id, source (S2/S3), country,
block_score, block_rank, block_top_score, prob, label, half (tune/report),
plus n_true per entity in reports/val_v2_truth.parquet (true-match counts
including pairs blocking missed).

All decision-rule and candidate-pruning experiments then run on this file
in seconds instead of re-blocking/re-featurizing each time.

Run: python -u -m src._cache_val_v2
"""
import numpy as np
import pandas as pd

from . import config, io_utils, model, pipeline_v2, progress
from .parallel_utils import DEFAULT_N_JOBS, parallel_pool

N_VAL = 20_000
K, MAX_DF = 10, 50_000
OUT_DIR = config.PROJECT_ROOT / "code" / "business_entity_resolution" / "reports"


def main() -> None:
    rng = np.random.default_rng(config.RANDOM_SEED)
    s1_all = io_utils.load_source(config.TRAIN_SOURCE1)
    s2 = io_utils.load_source(config.TRAIN_SOURCE2)
    s3 = io_utils.load_source(config.TRAIN_SOURCE3)
    all_true = io_utils.build_true_matches(io_utils.load_ground_truth())

    # identical validation-set construction to train_v2.py
    ck_idx = rng.choice(len(s1_all), size=50_000, replace=False)
    excluded = set(s1_all.iloc[ck_idx]["entity_id"])
    rest = s1_all[~s1_all["entity_id"].isin(excluded)]
    val = rest.iloc[np.random.default_rng(123).choice(len(rest), size=N_VAL, replace=False)].reset_index(drop=True)
    del s1_all, rest

    with parallel_pool(DEFAULT_N_JOBS) as pool:
        cands = pipeline_v2.generate_candidates(val, s2, s3, k=K, max_df=MAX_DF, pool=pool)
        X = pipeline_v2.featurize(cands, val, s2, s3, pool=pool)

    booster = model.load_model(config.MODEL_PATH.parent / "model_v2.txt")
    cands["prob"] = model.predict_proba(booster, X).astype(np.float32)
    cands["label"] = [int(b in all_true.get(a, ())) for a, b in
                      zip(cands["source1_entity_id"], cands["candidate_entity_id"])]
    cands["source"] = cands["candidate_entity_id"].str[:2]
    country = dict(zip(val["entity_id"], val["country"]))
    cands["country"] = cands["source1_entity_id"].map(country)

    # fixed tune/report halves (entity level)
    ids = val["entity_id"].to_numpy()
    tune_ids = set(np.random.default_rng(99).permutation(ids)[: N_VAL // 2])
    cands["half"] = np.where(cands["source1_entity_id"].isin(tune_ids), "tune", "report")
    truth = pd.DataFrame({
        "source1_entity_id": ids,
        "country": [country[e] for e in ids],
        "n_true": [len(all_true.get(e, ())) for e in ids],
        "true_ids": [",".join(sorted(all_true.get(e, ()))) for e in ids],
        "half": ["tune" if e in tune_ids else "report" for e in ids],
    })
    cands.to_parquet(OUT_DIR / "val_v2.parquet", index=False)
    truth.to_parquet(OUT_DIR / "val_v2_truth.parquet", index=False)
    progress.log(f"cached {len(cands)} candidate rows, {len(truth)} entities; "
                 f"labels {int(cands['label'].sum())}/{int(truth['n_true'].sum())} true pairs in candidates")
    progress.log("DONE")


if __name__ == "__main__":
    main()
