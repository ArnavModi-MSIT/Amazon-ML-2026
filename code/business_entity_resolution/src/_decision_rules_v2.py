"""Phase-1 experiments on the cached v2 validation set (reports/val_v2.parquet).
Post-processing only -- no re-blocking, no re-scoring.

Every rule is tuned on the "tune" half (10k entities) and reported on the
untouched "report" half (10k), so reported gains aren't threshold overfit.

Rules compared:
  single     predict every candidate with prob >= tau (what the current
             tau0/tau/tau_fallback = 0.3/0.7/0.7 collapses to)
  margin     empty if p1 < tau0; all p >= tau_all if p1 >= tau_all; else the
             best alone if p1 >= tau_fb and p1 - p2 >= margin
  optF       per entity, the prediction size k in {0..m} (top-k by prob)
             maximizing plug-in expected F0.5:
               E[F(k)] ~ 1.25 * sum_{i<=k} p_i / (0.25 * mu + k),  k >= 1
               E[F(0)] ~ w * prod_i (1 - p_i)          (entity is a singleton)
             with mu = c * sum_i p_i (expected number of true matches).
  optF/country  optF with (c, w) tuned separately per country.

Then a candidate-pruning Pareto table (candidates/entity vs F0.5) using the
best rule: fixed K per source, relative score floor, absolute score gap.

Run: python -u -m src._decision_rules_v2
"""
import itertools

import numpy as np
import pandas as pd

from . import config, evaluate, postprocess

REP = config.PROJECT_ROOT / "code" / "business_entity_resolution" / "reports"


def _truth(truth: pd.DataFrame) -> dict[str, set[str]]:
    return {e: set(t.split(",")) if t else set() for e, t in zip(truth["source1_entity_id"], truth["true_ids"])}


def _groups(c: pd.DataFrame):
    c = c.sort_values(["source1_entity_id", "prob"], ascending=[True, False])
    return {e: (g["candidate_entity_id"].to_numpy(), g["prob"].to_numpy())
            for e, g in c.groupby("source1_entity_id", sort=False)}


def rule_single(groups, ids, tau):
    return {e: set(groups[e][0][groups[e][1] >= tau]) if e in groups else set() for e in ids}


def rule_margin(groups, ids, tau0, tau_all, tau_fb, margin):
    out = {}
    for e in ids:
        if e not in groups:
            out[e] = set()
            continue
        cids, p = groups[e]
        p1 = p[0]
        p2 = p[1] if len(p) > 1 else 0.0
        if p1 < tau0:
            out[e] = set()
        elif p1 >= tau_all:
            out[e] = set(cids[p >= tau_all])
        elif p1 >= tau_fb and p1 - p2 >= margin:
            out[e] = {cids[0]}
        else:
            out[e] = set()
    return out


def rule_optf(groups, ids, c, w):
    out = {}
    for e in ids:
        if e not in groups:
            out[e] = set()
            continue
        cids, p = groups[e]
        mu = c * p.sum()
        k = np.arange(1, len(p) + 1)
        ef = 1.25 * np.cumsum(p) / (0.25 * mu + k)
        e0 = w * np.prod(1.0 - p)
        best = int(np.argmax(ef))
        out[e] = set(cids[: best + 1]) if ef[best] > e0 else set()
    return out


def score(preds, cands, truth_map, ids):
    preds = postprocess.resolve_conflicts(preds, cands[["source1_entity_id", "candidate_entity_id", "prob"]])
    rep = evaluate.score_report(truth_map, preds, ids)
    n = sum(len(v) for v in preds.values())
    return rep["macro_f0.5"], rep["singleton_macro_f0.5"], rep["nonsingleton_macro_f0.5"], n / len(ids)


def tune(rule, grid, groups, cands, truth_map, ids):
    best = None
    for params in grid:
        f = score(rule(groups, ids, *params), cands, truth_map, ids)[0]
        if best is None or f > best[0]:
            best = (f, params)
    return best


def main() -> None:
    cands = pd.read_parquet(REP / "val_v2.parquet")
    truth = pd.read_parquet(REP / "val_v2_truth.parquet")
    tm = _truth(truth)
    halves = {h: truth.loc[truth["half"] == h, "source1_entity_id"].tolist() for h in ("tune", "report")}
    ch = {h: cands[cands["half"] == h] for h in halves}
    gh = {h: _groups(ch[h]) for h in halves}
    country = dict(zip(truth["source1_entity_id"], truth["country"]))

    # ---- loss decomposition (report half, current rule) ----
    rep_ids = halves["report"]
    n_true = sum(len(tm[e]) for e in rep_ids)
    in_cand = int(ch["report"]["label"].sum())
    cur = postprocess.resolve_conflicts(rule_single(gh["report"], rep_ids, 0.7),
                                        ch["report"][["source1_entity_id", "candidate_entity_id", "prob"]])
    tp = sum(len(cur[e] & tm[e]) for e in rep_ids)
    fp = sum(len(cur[e] - tm[e]) for e in rep_ids)
    sing_fp = sum(1 for e in rep_ids if not tm[e] and cur[e])
    zeroed = sum(1 for e in rep_ids if tm[e] and not cur[e])
    print("=== Loss decomposition (report half, current rule = prob >= 0.7) ===")
    print(f"true pairs {n_true}; in candidates {in_cand} ({in_cand/n_true:.3f}); "
          f"predicted TP {tp} ({tp/n_true:.3f} of truth, {tp/in_cand:.3f} of reachable); FP {fp}; "
          f"precision {tp/(tp+fp):.3f}")
    print(f"singletons with >=1 FP: {sing_fp}/{sum(1 for e in rep_ids if not tm[e])}; "
          f"non-singletons predicted empty: {zeroed}/{sum(1 for e in rep_ids if tm[e])}")

    # ---- decision rules ----
    rows = []
    t_ids, r_ids = halves["tune"], halves["report"]
    for name, rule, grid in (
        ("single", rule_single, [(t,) for t in np.arange(0.40, 0.91, 0.05).round(2)]),
        ("margin", rule_margin, list(itertools.product([0.2, 0.3, 0.4], np.arange(0.55, 0.86, 0.05).round(2),
                                                       [0.4, 0.5, 0.6], [0.0, 0.1, 0.2, 0.3]))),
        ("optF", rule_optf, list(itertools.product([0.8, 0.9, 1.0, 1.1, 1.2, 1.4], [0.6, 0.8, 1.0, 1.2, 1.5]))),
    ):
        f_t, params = tune(rule, grid, gh["tune"], ch["tune"], tm, t_ids)
        f_r, sing, nons, ppe = score(rule(gh["report"], r_ids, *params), ch["report"], tm, r_ids)
        rows.append((name, str(params), f_t, f_r, sing, nons, ppe))
        if name == "optF":
            best_optf = params

    # per-country optF
    per_c = {}
    for cty in sorted(set(country.values())):
        ids_c = [e for e in t_ids if country[e] == cty]
        per_c[cty] = tune(rule_optf, list(itertools.product([0.8, 0.9, 1.0, 1.1, 1.2, 1.4], [0.6, 0.8, 1.0, 1.2, 1.5])),
                          gh["tune"], ch["tune"], tm, ids_c)[1]
    preds = {}
    for cty, params in per_c.items():
        preds.update(rule_optf(gh["report"], [e for e in r_ids if country[e] == cty], *params))
    f_r, sing, nons, ppe = score(preds, ch["report"], tm, r_ids)
    rows.append(("optF/country", str(per_c), float("nan"), f_r, sing, nons, ppe))

    print("\n=== Decision rules (tuned on 'tune', reported on 'report') ===")
    print(pd.DataFrame(rows, columns=["rule", "params", "F_tune", "F_report", "singleton", "non-singleton",
                                      "pred/entity"]).to_string(index=False))
    for cty in sorted(set(country.values())):
        ids_c = [e for e in r_ids if country[e] == cty]
        f_single = score(rule_single(gh["report"], ids_c, 0.7), ch["report"], tm, ids_c)[0]
        f_opt = score(rule_optf(gh["report"], ids_c, *best_optf), ch["report"], tm, ids_c)[0]
        print(f"  {cty}: single@0.7 {f_single:.4f} -> optF {f_opt:.4f}")

    # ---- candidate pruning Pareto (single-threshold rule, tau re-tuned per policy on 'tune') ----
    print("\n=== Candidate pruning (single threshold, tau tuned per policy on 'tune', reported on 'report') ===")

    def masks(c):
        top = c.groupby(["source1_entity_id", "source"])["block_score"].transform("max")
        m = [(f"K={k}/source", c["block_rank"] < k) for k in (3, 4, 5, 6, 7, 8, 10)]
        m += [(f"K_S2={a},K_S3={b}", ((c["source"] == "S2") & (c["block_rank"] < a)) |
               ((c["source"] == "S3") & (c["block_rank"] < b))) for a, b in ((5, 6), (6, 5), (4, 6))]
        m += [(f"rel floor {b}*top (min 2)", (c["block_rank"] < 2) | (c["block_score"] >= b * top))
              for b in (0.4, 0.5, 0.6)]
        m += [(f"rel floor {b}*top (min 3)", (c["block_rank"] < 3) | (c["block_score"] >= b * top))
              for b in (0.5, 0.6)]
        return m

    n_true_rep = n_true
    prow = []
    for (name, mt), (_, mr) in zip(masks(ch["tune"]), masks(ch["report"])):
        st, sr = ch["tune"][mt], ch["report"][mr]
        tau = tune(rule_single, [(t,) for t in np.arange(0.50, 0.86, 0.05).round(2)],
                   _groups(st), st, tm, t_ids)[1]
        f, sing, nons, ppe = score(rule_single(_groups(sr), r_ids, *tau), sr, tm, r_ids)
        prow.append((name, len(sr) / len(r_ids), sr["label"].sum() / n_true_rep, tau[0], f, sing, nons))
    print(pd.DataFrame(prow, columns=["policy", "cand/entity", "recall_ceiling", "tau", "F_report", "singleton",
                                      "non-singleton"]).to_string(index=False))


if __name__ == "__main__":
    main()
