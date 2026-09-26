"""Inference on the TEST split -> output/matching_results.tsv and
output/candidate_pairs.tsv, then the official validator.

Usage (from code/business_entity_resolution/):
    python -u -m src.run_inference_v2 --tag v4

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

from . import config, io_utils, model, pipeline_v2, postprocess, progress, text_repr
from .parallel_utils import DEFAULT_N_JOBS, parallel_pool

BATCH = 100_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4")
    args = ap.parse_args()

    with open(config.THRESHOLDS_PATH.parent / f"thresholds_{args.tag}.json") as f:
        th = json.load(f)
    # a model trained on another text representation would get silently shifted features
    repr_version = th.get("repr_version", "v3")
    if repr_version != text_repr.REPR_VERSION:
        sys.exit(f"model_{args.tag} was trained on text representation {repr_version}, this code is "
                 f"{text_repr.REPR_VERSION}; retrain, or use the code that produced it "
                 f"(v3 = submission 04, in the GitHub repo)")
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
    # every scored candidate, so thresholds / per-country checks never need another 3 h run
    score_dir = config.ARTIFACTS_DIR / f"test_scores_{args.tag}"
    score_dir.mkdir(exist_ok=True)
    with parallel_pool(DEFAULT_N_JOBS) as pool:
        fw = th.get("field_weights")
        cands = pipeline_v2.generate_candidates(s1, s2, s3, k=k, max_df=th["max_df"], pool=pool,
                                                field_weights={c: tuple(w) for c, w in fw.items()} if fw else None,
                                                segment=th.get("segment", False))
        cands = pipeline_v2.prune_candidates(cands, th.get("rel_floor"), th.get("min_rank", 3),
                                             country_of=pd.Series(s1["country"].to_numpy(), index=s1["entity_id"]))
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
            ps.to_parquet(score_dir / f"batch_{i // BATCH:02d}.parquet", index=False)
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
    n_before = sum(len(v) for v in preds.values())
    preds = postprocess.resolve_conflicts(preds, scores)
    n_after = sum(len(v) for v in preds.values())
    progress.log(f"Conflict resolution removed {n_before - n_after} of {n_before} predicted pairs "
                 f"({(n_before - n_after) / max(n_before, 1):.2%}); validation only sees 10k S1 competing")

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
    country = dict(zip(s1["entity_id"], s1["country"]))
    top1 = pd.concat([pd.read_parquet(f) for f in sorted(score_dir.glob("batch_*.parquet"))]) \
        .groupby("source1_entity_id")["prob"].max()
    stats = pd.DataFrame({"country": [country[e] for e in s1_ids],
                          "cands": [len(cand_sets.get(e, ())) for e in s1_ids],
                          "preds": [len(preds.get(e, ())) for e in s1_ids],
                          "top1": top1.reindex(s1_ids).fillna(0).to_numpy()})
    progress.log("Per-country (val India/US: ~3.1 predicted/entity, ~6% empty):\n" + stats.groupby("country").agg(
        cands_per_entity=("cands", "mean"), preds_per_entity=("preds", "mean"),
        empty_rate=("preds", lambda x: (x == 0).mean()), mean_top1_prob=("top1", "mean")).round(3).to_string())
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
