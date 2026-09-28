"""Train + validate at REAL pool scale.

Train entities: 60k train S1 (54k fit / 6k early-stopping). Validation: a
fixed 20k held-out set, split into fixed "tune" / "report" halves (10k each):
the decision threshold is tuned on "tune" and the headline F0.5 is measured
on "report" (so it isn't threshold-overfit). The v1 pipeline scored 0.486 on
this set, tracking its 0.519 leaderboard score. All blocking is against the
FULL train S2/S3 pools, with the same candidate policy used at inference.

Writes artifacts/model_<tag>.txt, artifacts/thresholds_<tag>.json and
reports/val_<tag>.parquet (scored validation candidates).

Run: python -u -m src.train_v2 --tag v4
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import config, evaluate, io_utils, model, pipeline_v2, postprocess, progress
from . import text_repr
from .parallel_utils import DEFAULT_N_JOBS, parallel_pool

N_TRAIN, N_ES, N_VAL = 54_000, 6_000, 20_000
K, MAX_DF = 10, 50_000
TAU_GRID = np.arange(0.50, 0.951, 0.01).round(2)  # fine: the optimum is flat and split noise ~0.001


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
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--rel-floor", type=float, default=0.5, help="none = no pruning")
    ap.add_argument("--min-rank", type=int, default=3)
    ap.add_argument("--country-floor", default=None, help='JSON {country: rel_floor} overriding --rel-floor')
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--n-train", type=int, default=N_TRAIN, help="fit entities (54k while iterating)")
    ap.add_argument("--segment", action="store_true", help="segment glued pool names in blocking")
    ap.add_argument("--n-ensemble", type=int, default=1, help="seed-bagged LightGBM members (probabilities averaged)")
    ap.add_argument("--latin-only-tokens", action="store_true", help="drop non-ASCII words from blocking vectors")
    ap.add_argument("--translit", action="store_true", help="learned native-script word dictionary (translit_dict)")
    ap.add_argument("--key-channel", action="store_true", help="exact name+number key candidates next to top-K")
    ap.add_argument("--reverse", action="store_true", help="reverse top-1 channel: each pool record's best S1 entity")
    ap.add_argument("--lgb-params", default=None, help='JSON LightGBM overrides, e.g. {"num_leaves": 127}')
    ap.add_argument("--universe-frac", type=float, default=1.0,
                    help="keep this fraction of the train S1 file as the 'S1 universe' (reverse pass, S1 counts); the "
                         "fit/es/val entities are always kept. Test has ~40%% ownerless pool records vs ~26%% in train; "
                         "0.81 reproduces the test ratio (S1/pool 1.73M/9.97M = 0.174)")
    ap.add_argument("--stage1-recall", type=float, default=0.0,
                    help="supervised meta-blocking: keep this share of true pairs (e.g. 0.999) with a blocking-signal "
                         "LightGBM before the matcher; 0 = off")
    ap.add_argument("--dump-pairs", action="store_true",
                    help="save every pruned candidate pair (label, role, stage-1 score) to reports/pairs_<tag>.parquet")
    ap.add_argument("--empty-k", type=int, default=0, help="empty-address channel: top-k among empty-address pool rows")
    ap.add_argument("--compare-models", default=None,
                    help="comma-separated earlier tags (e.g. v9,v10): score THIS run's validation features with those "
                         "saved models too (each on its own feature subset), so models are compared on identical rows")
    ap.add_argument("--field-weights", default=None,
                    help='JSON {country: [w_name, w_addr, empty_addr_c]}: separate name/address '
                         'normalization in blocking for those countries (default: one joint cosine)')
    args = ap.parse_args()
    rel_floor = None if args.no_prune else args.rel_floor
    if rel_floor is not None and args.country_floor:
        rel_floor = {"default": rel_floor, **json.loads(args.country_floor)}
    field_weights = {c: tuple(w) for c, w in json.loads(args.field_weights).items()} if args.field_weights else None

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
    tr = remaining.iloc[np.random.default_rng(7).choice(len(remaining), size=args.n_train + N_ES, replace=False)]
    fit_ids = set(tr["entity_id"].iloc[:args.n_train])
    es_ids = set(tr["entity_id"].iloc[args.n_train:])
    s1 = pd.concat([tr, val], ignore_index=True)
    universe = s1_all
    if args.universe_frac < 1.0:  # test-like decoy density: drop other S1 entities, their pool records lose their owner
        others = s1_all[~s1_all["entity_id"].isin(set(s1["entity_id"]))]
        n_keep = max(int(args.universe_frac * len(s1_all)) - len(s1), 0)
        universe = pd.concat([s1, others.sample(n=min(n_keep, len(others)), random_state=11)], ignore_index=True)
        progress.log(f"S1 universe: {len(universe)} of {len(s1_all)} train S1 entities (test-like decoy density)")
    del rest, remaining  # s1_all stays: S1-frequency features are counted over the whole S1 file

    translit_file = None
    if args.translit:  # before the pool starts, so worker processes load the same dictionary
        from . import translit_dict
        mapping = translit_dict.build(s1_all, [s2, s3], all_true, exclude_s1=set(val_ids))
        translit_file = f"translit_{args.tag}.json"
        translit_dict.save(mapping, config.ARTIFACTS_DIR / translit_file)
        translit_dict.activate(config.ARTIFACTS_DIR / translit_file)
        progress.log(f"translit dictionary: {len(mapping)} words (validation entities excluded)")

    with parallel_pool(DEFAULT_N_JOBS) as pool:
        cands = pipeline_v2.generate_candidates(s1, s2, s3, k=K, max_df=MAX_DF, pool=pool, s1_universe=universe,
                                                field_weights=field_weights, segment=args.segment,
                                                latin_only_tokens=args.latin_only_tokens, key_channel=args.key_channel, empty_k=args.empty_k,
                                                reverse=args.reverse)
        del s1_all, universe
        cands = pipeline_v2.prune_candidates(cands, rel_floor, args.min_rank,
                                             country_of=pd.Series(s1["country"].to_numpy(), index=s1["entity_id"]))
        cands["label"] = [
            int(b in all_true.get(a, ())) for a, b in zip(cands["source1_entity_id"], cands["candidate_entity_id"])
        ]
        for nm, ids in (("fit", fit_ids), ("val", set(val_ids))):
            n_true = sum(len(all_true.get(e, ())) for e in ids)
            sub = cands["source1_entity_id"].isin(ids)
            progress.log(f"[{nm}] recall ceiling {cands.loc[sub, 'label'].sum() / n_true:.4f}, "
                         f"{sub.sum() / len(ids):.2f} candidates/entity")
        stage1 = None
        if args.stage1_recall:
            from . import meta_blocking
            cty_of = pd.Series(s1["country"].to_numpy(), index=s1["entity_id"])
            X1 = meta_blocking.stage1_matrix(cands, cty_of, (field_weights or {}).keys())
            y1 = cands["label"].to_numpy()
            fit_m = cands["source1_entity_id"].isin(fit_ids).to_numpy()
            st1, tau1, oof = meta_blocking.fit_with_oof(X1, y1, cands["source1_entity_id"], fit_m, args.stage1_recall)
            p1 = oof.copy()
            p1[~fit_m] = st1.predict(X1[~fit_m])
            keep = p1 >= tau1
            if args.dump_pairs:
                role = np.where(fit_m, "fit", np.where(cands["source1_entity_id"].isin(es_ids).to_numpy(), "es",
                                np.where(cands["source1_entity_id"].isin(tune_ids).to_numpy(), "tune", "report")))
                pd.DataFrame({"source1_entity_id": cands["source1_entity_id"].to_numpy(),
                              "candidate_entity_id": cands["candidate_entity_id"].to_numpy(),
                              "source": cands["source"].to_numpy(), "label": y1, "p1": p1, "role": role})                     .to_parquet(config.PROJECT_ROOT / "code" / "business_entity_resolution" / "reports" /
                                f"pairs_{args.tag}.parquet", index=False)
                progress.log(f"dumped {len(cands)} pre-filter pairs to pairs_{args.tag}.parquet")
            for nm, ids in (("fit", fit_ids), ("val", set(val_ids))):
                n_true = sum(len(all_true.get(e, ())) for e in ids)
                sub = cands["source1_entity_id"].isin(ids).to_numpy()
                progress.log(f"[stage1 {nm}] tau {tau1:.5f}: keep {keep[sub].mean():.1%} of candidates | recall ceiling "
                             f"{y1[sub & keep].sum() / n_true:.4f} | {(sub & keep).sum() / len(ids):.2f} candidates/entity")
            st1.save_model(str(config.ARTIFACTS_DIR / f"stage1_{args.tag}.txt"))
            stage1 = {"stage1_model": f"stage1_{args.tag}.txt", "stage1_tau": tau1, "stage1_recall": args.stage1_recall}
            cands = cands[keep].reset_index(drop=True)
            del X1, oof, p1, keep
        progress.log("featurizing...")
        X = pipeline_v2.featurize(cands, s1, s2, s3, pool=pool)[pipeline_v2.FEATURE_NAMES_ALL]

    y = cands["label"].to_numpy()
    is_fit = cands["source1_entity_id"].isin(fit_ids).to_numpy()
    is_es = cands["source1_entity_id"].isin(es_ids).to_numpy()
    is_val = cands["source1_entity_id"].isin(set(val_ids)).to_numpy()
    progress.log(f"train rows {is_fit.sum()} (pos {y[is_fit].sum()}), es rows {is_es.sum()}, val rows {is_val.sum()}")

    lgb_params = json.loads(args.lgb_params) if args.lgb_params else {}
    boosters = model.train_ensemble(X[is_fit], y[is_fit], X[is_es], y[is_es], params=lgb_params, num_boost_round=5000,
                                    early_stopping_rounds=50, n_members=args.n_ensemble)
    booster = boosters[0]
    progress.log(f"trained {len(boosters)} member(s), best_iteration={[b.best_iteration for b in boosters]}")
    progress.log("top features:\n" + model.feature_importance(booster).head(15).to_string())

    vs = cands.loc[is_val].copy()
    vs["prob"] = model.predict_proba_ensemble(boosters, X[is_val]).astype(np.float32)
    if len(boosters) > 1:  # single-member score on the same rows, to see what the ensemble adds
        vs["prob_member0"] = model.predict_proba(booster, X[is_val]).astype(np.float32)
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

    for old_tag in (args.compare_models.split(",") if args.compare_models else []):
        with open(config.THRESHOLDS_PATH.parent / f"thresholds_{old_tag}.json") as f:
            old_n = json.load(f).get("n_ensemble", 1)
        old = model.load_models(config.MODEL_PATH.parent / f"model_{old_tag}", old_n)
        own = vs["prob"].to_numpy().copy()
        p_old = model.predict_proba_ensemble(old, X.loc[is_val, old[0].feature_name()]).astype(np.float32)
        vs["prob"] = p_old  # _f05 reads "prob"
        t_old = max(TAU_GRID, key=lambda t: _f05(vs[vs["half"] == "tune"], truth, t_ids, t)[0]["macro_f0.5"])
        r_old = _f05(vs[vs["half"] == "report"], truth, r_ids, t_old)[0]["macro_f0.5"]
        per_c = ", ".join(f"{c} {_f05(vs[vs['half'] == 'report'], truth, [e for e in r_ids if country[e] == c], t_old)[0]['macro_f0.5']:.4f}"
                          for c in sorted(set(country.values())))
        progress.log(f"[compare] model_{old_tag} on these validation rows: tau={t_old} | REPORT-half macro F0.5 = "
                     f"{r_old:.4f} ({per_c})")
        vs[f"prob_{old_tag}"] = p_old
        vs["prob"] = own

    model.save_models(boosters, config.MODEL_PATH.parent / f"model_{args.tag}")
    with open(config.THRESHOLDS_PATH.parent / f"thresholds_{args.tag}.json", "w") as f:
        json.dump({"tau0": 0.0, "tau": float(best_tau), "tau_fallback": float(best_tau), "k": K, "max_df": MAX_DF,
                   "rel_floor": rel_floor, "min_rank": args.min_rank, "repr_version": text_repr.REPR_VERSION,
                   "field_weights": field_weights, "segment": args.segment,
                   "n_ensemble": len(boosters), "latin_only_tokens": args.latin_only_tokens,
                   "translit": translit_file, "key_channel": args.key_channel, "reverse": args.reverse,
                   "universe_frac": args.universe_frac, "empty_k": args.empty_k, **(stage1 or {}), "lgb_params": lgb_params}, f, indent=2)
    vs.to_parquet(config.PROJECT_ROOT / "code" / "business_entity_resolution" / "reports" / f"val_{args.tag}.parquet",
                  index=False)
    progress.log(f"saved model_{args.tag}.txt, thresholds_{args.tag}.json, val_{args.tag}.parquet")
    progress.log("DONE")


if __name__ == "__main__":
    main()
