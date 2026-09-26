"""Train + validate at REAL pool scale.

Train entities: 60k train S1 (54k fit / 6k early-stopping). Validation: a
fixed 20k held-out set, split into fixed "tune" / "report" halves (10k each):
the decision threshold is tuned on "tune" and the headline F0.5 is measured
on "report" (so it isn't threshold-overfit). The v1 pipeline scored 0.486 on
this set, tracking its 0.519 leaderboard score. All blocking is against the
FULL train S2/S3 pools, with the same candidate policy used at inference.

Writes artifacts/model_<tag>.txt, artifacts/thresholds_<tag>.json and
reports/val_<tag>.parquet (scored validation candidates).

Run: python -u -m src.train_v2 --tag v3
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import config, evaluate, io_utils, model, pipeline_v2, postprocess, progress
from .parallel_utils import DEFAULT_N_JOBS, parallel_pool

N_TRAIN, N_ES, N_VAL = 54_000, 6_000, 20_000
K, MAX_DF = 10, 50_000
TAU_GRID = np.arange(0.50, 0.86, 0.05).round(2)


def _f05(scores: pd.DataFrame, truth: dict, ids: list, tau: float):
    s = scores[scores["prob"] >= tau]
    preds = {e: set() for e in ids}
    for a, b in zip(s["source1_entity_id"], s["candidate_entity_id"]):
        if a in preds:
            preds[a].add(b)
    preds = postprocess.resolve_conflicts(preds, scores)
    return evaluate.score_report(truth, preds, ids), sum(len(v) for v in preds.values()) / len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--rel-floor", type=float, default=0.5, help="none = no pruning")
    ap.add_argument("--min-rank", type=int, default=3)
    ap.add_argument("--no-prune", action="store_true")
    args = ap.parse_args()
    rel_floor = None if args.no_prune else args.rel_floor

    rng = np.random.default_rng(config.RANDOM_SEED)
    progress.log("Loading train data...")
    s1_all = io_utils.load_source(config.TRAIN_SOURCE1)
    s2 = io_utils.load_source(config.TRAIN_SOURCE2)
    s3 = io_utils.load_source(config.TRAIN_SOURCE3)
    all_true = io_utils.build_true_matches(io_utils.load_ground_truth())

    # fixed validation set (seeded) -- the same one v1 was measured on
    ck_idx = rng.choice(len(s1_all), size=50_000, replace=False)
    excluded = set(s1_all.iloc[ck_idx]["entity_id"])
    rest = s1_all[~s1_all["entity_id"].isin(excluded)]
    val = rest.iloc[np.random.default_rng(123).choice(len(rest), size=N_VAL, replace=False)]
    val_ids = list(val["entity_id"])
    tune_ids = set(np.random.default_rng(99).permutation(np.array(val_ids))[: N_VAL // 2])
    remaining = s1_all[~s1_all["entity_id"].isin(set(val_ids))]
    tr = remaining.iloc[np.random.default_rng(7).choice(len(remaining), size=N_TRAIN + N_ES, replace=False)]
    fit_ids = set(tr["entity_id"].iloc[:N_TRAIN])
    es_ids = set(tr["entity_id"].iloc[N_TRAIN:])
    s1 = pd.concat([tr, val], ignore_index=True)
    del s1_all, rest, remaining

    with parallel_pool(DEFAULT_N_JOBS) as pool:
        cands = pipeline_v2.generate_candidates(s1, s2, s3, k=K, max_df=MAX_DF, pool=pool)
        cands = pipeline_v2.prune_candidates(cands, rel_floor, args.min_rank)
        cands["label"] = [
            int(b in all_true.get(a, ())) for a, b in zip(cands["source1_entity_id"], cands["candidate_entity_id"])
        ]
        for nm, ids in (("fit", fit_ids), ("val", set(val_ids))):
            n_true = sum(len(all_true.get(e, ())) for e in ids)
            sub = cands["source1_entity_id"].isin(ids)
            progress.log(f"[{nm}] recall ceiling {cands.loc[sub, 'label'].sum() / n_true:.4f}, "
                         f"{sub.sum() / len(ids):.2f} candidates/entity")
        progress.log("featurizing...")
        X = pipeline_v2.featurize(cands, s1, s2, s3, pool=pool)[pipeline_v2.FEATURE_NAMES_V3]

    y = cands["label"].to_numpy()
    is_fit = cands["source1_entity_id"].isin(fit_ids).to_numpy()
    is_es = cands["source1_entity_id"].isin(es_ids).to_numpy()
    is_val = cands["source1_entity_id"].isin(set(val_ids)).to_numpy()
    progress.log(f"train rows {is_fit.sum()} (pos {y[is_fit].sum()}), es rows {is_es.sum()}, val rows {is_val.sum()}")

    booster = model.train(X[is_fit], y[is_fit], X[is_es], y[is_es], num_boost_round=2000, early_stopping_rounds=50)
    progress.log(f"trained, best_iteration={booster.best_iteration}")
    progress.log("top features:\n" + model.feature_importance(booster).head(15).to_string())

    vs = cands.loc[is_val].copy()
    vs["prob"] = model.predict_proba(booster, X[is_val]).astype(np.float32)
    vs["half"] = np.where(vs["source1_entity_id"].isin(tune_ids), "tune", "report")
    truth = {e: all_true.get(e, set()) for e in val_ids}
    t_ids = [e for e in val_ids if e in tune_ids]
    r_ids = [e for e in val_ids if e not in tune_ids]
    best_tau = max(TAU_GRID, key=lambda t: _f05(vs[vs["half"] == "tune"], truth, t_ids, t)[0]["macro_f0.5"])
    rep, ppe = _f05(vs[vs["half"] == "report"], truth, r_ids, best_tau)
    progress.log(f"[{args.tag}] tau={best_tau} (tuned on tune half) | REPORT-half macro F0.5 = {rep['macro_f0.5']:.4f} | "
                 f"singleton {rep['singleton_macro_f0.5']:.4f} | non-singleton {rep['nonsingleton_macro_f0.5']:.4f} | "
                 f"{ppe:.2f} predicted/entity")
    country = dict(zip(val["entity_id"], val["country"]))
    for cty in sorted(set(country.values())):
        ids_c = [e for e in r_ids if country[e] == cty]
        progress.log(f"[{args.tag}]   {cty}: {_f05(vs[vs['half'] == 'report'], truth, ids_c, best_tau)[0]['macro_f0.5']:.4f}")

    model.save_model(booster, config.MODEL_PATH.parent / f"model_{args.tag}.txt")
    with open(config.THRESHOLDS_PATH.parent / f"thresholds_{args.tag}.json", "w") as f:
        json.dump({"tau0": 0.0, "tau": float(best_tau), "tau_fallback": float(best_tau), "k": K, "max_df": MAX_DF,
                   "rel_floor": rel_floor, "min_rank": args.min_rank}, f, indent=2)
    vs.to_parquet(config.PROJECT_ROOT / "code" / "business_entity_resolution" / "reports" / f"val_{args.tag}.parquet",
                  index=False)
    progress.log(f"saved model_{args.tag}.txt, thresholds_{args.tag}.json, val_{args.tag}.parquet")
    progress.log("DONE")


if __name__ == "__main__":
    main()
