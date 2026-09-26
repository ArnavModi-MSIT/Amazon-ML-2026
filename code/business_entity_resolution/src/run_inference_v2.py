"""Inference on the TEST split -> output/matching_results.tsv and
output/candidate_pairs.tsv, then the official validator.

Usage (from code/business_entity_resolution/):
    python -u -m src.run_inference_v2 --tag v3

Loads artifacts/model_<tag>.txt and artifacts/thresholds_<tag>.json; the
candidate policy (K, max_df, relative-floor pruning) comes from the JSON so
it is exactly the policy the model was trained and validated with, and the
feature columns come from the model file itself.

Candidate pairs are generated for all S1 entities at once (IDF top-K), then
features/predictions run in S1 batches to bound memory. Thresholds are
applied per entity inside each batch; the one-to-one constraint (an S2/S3
record matches at most one S1 entity -- verified on train ground truth) is
enforced once, globally, after all batches.
"""
import argparse
import json
import subprocess
import sys
import time

import pandas as pd

from . import config, io_utils, model, pipeline_v2, postprocess, progress
from .parallel_utils import DEFAULT_N_JOBS, parallel_pool

BATCH = 100_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args()

    with open(config.THRESHOLDS_PATH.parent / f"thresholds_{args.tag}.json") as f:
        th = json.load(f)
    k = th["k"]
    booster = model.load_model(config.MODEL_PATH.parent / f"model_{args.tag}.txt")
    feat_cols = booster.feature_name()
    t0 = time.time()

    progress.log("Loading TEST data...")
    s1 = io_utils.load_source(config.TEST_SOURCE1)
    s2 = io_utils.load_source(config.TEST_SOURCE2)
    s3 = io_utils.load_source(config.TEST_SOURCE3)
    progress.log(f"S1 {len(s1)}, S2 {len(s2)}, S3 {len(s3)}; K={k}, max_df={th['max_df']}")

    matched = []
    with parallel_pool(DEFAULT_N_JOBS) as pool:
        cands = pipeline_v2.generate_candidates(s1, s2, s3, k=k, max_df=th["max_df"], pool=pool)
        cands = pipeline_v2.prune_candidates(cands, th.get("rel_floor"), th.get("min_rank", 3))
        cands = cands.sort_values("source1_entity_id", kind="stable").reset_index(drop=True)
        s1_ids = s1["entity_id"].tolist()
        for i in range(0, len(s1_ids), BATCH):
            ids = s1_ids[i:i + BATCH]
            sub = cands[cands["source1_entity_id"].isin(set(ids))].reset_index(drop=True)
            if sub.empty:
                continue
            X = pipeline_v2.featurize(sub, s1[s1["entity_id"].isin(set(ids))], s2, s3, pool=pool)[feat_cols]
            ps = sub[["source1_entity_id", "candidate_entity_id"]].copy()
            ps["prob"] = model.predict_proba(booster, X)
            del X
            m = postprocess.apply_thresholds(ps, ids, th["tau0"], th["tau"], th["tau_fallback"])
            keep = {(a, b) for a, bs in m.items() for b in bs}
            ps = ps[[(a, b) in keep for a, b in zip(ps["source1_entity_id"], ps["candidate_entity_id"])]]
            matched.append(ps)
            progress.log(f"batch {i // BATCH + 1}/{-(-len(s1_ids) // BATCH)}: {len(sub)} pairs, "
                         f"{len(ps)} predicted matches")

    scores = pd.concat(matched, ignore_index=True)
    progress.log(f"Global conflict resolution over {len(scores)} predicted matches...")
    preds = {e: set() for e in s1_ids}
    for a, b in zip(scores["source1_entity_id"], scores["candidate_entity_id"]):
        preds[a].add(b)
    preds = postprocess.resolve_conflicts(preds, scores)

    cand_sets = cands.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()
    match_rows, cand_rows = [], []
    for e in s1_ids:
        c = cand_sets.get(e, set())
        mt = preds.get(e, set())
        assert mt <= c, f"{e}: match outside candidate set"
        match_rows.append((e, ",".join(sorted(mt))))
        cand_rows.append((e, ",".join(sorted(c))))
    io_utils.write_id_list_file(pd.DataFrame(match_rows, columns=config.MATCHING_RESULTS_COLUMNS),
                                config.MATCHING_RESULTS_PATH, config.MATCHING_RESULTS_COLUMNS)
    io_utils.write_id_list_file(pd.DataFrame(cand_rows, columns=config.CANDIDATE_PAIRS_COLUMNS),
                                config.CANDIDATE_PAIRS_PATH, config.CANDIDATE_PAIRS_COLUMNS)
    n_m = sum(1 for _, x in match_rows if x)
    n_pairs = sum(len(v) for v in preds.values())
    progress.log(f"Wrote outputs: {n_m}/{len(s1_ids)} entities matched, {n_pairs} pairs "
                 f"({n_pairs/len(s1_ids):.2f}/entity); {len(cands)/len(s1_ids):.2f} candidates/entity; "
                 f"total {time.time()-t0:.0f}s")

    r = subprocess.run([sys.executable, str(config.PROJECT_ROOT / "utils" / "validate_submission.py"),
                        "--matching", str(config.MATCHING_RESULTS_PATH),
                        "--candidate", str(config.CANDIDATE_PAIRS_PATH),
                        "--test-dir", str(config.TEST_DIR)], capture_output=True, text=True)
    progress.log(r.stdout.strip())
    progress.log(f"Validator exit code: {r.returncode} ({'PASS' if r.returncode == 0 else 'FAIL'})")


if __name__ == "__main__":
    main()
