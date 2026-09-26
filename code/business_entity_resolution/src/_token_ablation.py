"""Blocking ablation: recall of each blocking-token variant on the fixed 20k
held-out validation entities vs the FULL train pools (same reps, same K and
pruning policy as training/inference), so a token change is judged on the
recall ceiling before any retraining.

Run: python -u -m src._token_ablation
"""
import numpy as np
import pandas as pd

from . import blocking, blocking_idf, config, io_utils, pipeline_v2, progress
from . import text_repr as tr
from .parallel_utils import DEFAULT_N_JOBS, parallel_pool

N_VAL = 20_000


def _tokens(variant: set):
    def fn(name_norm, name_skeleton, addr_norm, phonetic, name_seg=""):
        toks = {"n:" + t for t in name_norm.split() if len(t) >= 2}
        toks |= {"n:" + t for t in name_skeleton.split() if len(t) >= 2}
        toks |= {"a:" + t for t in addr_norm.split() if len(t) >= 2 or ("digits" in variant and t.isdigit())}
        if "j" in variant:
            joined = "".join(t for t in name_norm.split() if t not in tr.LEGAL_SUFFIXES | {"and", "the", "of"})
            if len(joined) >= 4:
                toks.add("j:" + joined)
        if "p" in variant and (phonetic or "pall" in variant):
            toks |= {"p:" + c for c in tr.phonetic_tokens(name_skeleton)}
        if "seg" in variant and name_seg:
            toks |= {"n:" + t for t in name_seg.split() if len(t) >= 2}
        return list(toks)
    return fn


V4 = {"digits", "p"}
# name: (token variant, field_weights) -- first round measured token variants
# (v4 = digits + p); second round: separate name/address normalization.
# Round 2 (pruned recall): joint 0.9402 | split 1/1 0.9231 | 1/.7 0.8825 | .7/1 0.9326 | .5/1 0.9252.
# Round 3: empty pool address counted as partial agreement c (third value).
# Round 3: 1/1 c.5 0.9227 (India .9309, US .9173) | .7/1 c.5 0.9433 (India .9349, US .9488, 13.6 cand)
# Round 4: per-country weights (India split, US joint) + glued-name segmentation; phonetic on all pool records.
IN = {"India": (0.7, 1.0, 0.5)}
VARIANTS = {
    "joint+seg": (V4 | {"seg"}, None),
    "IN.7c.5": (V4, IN),
    "IN.7c.5+seg": (V4 | {"seg"}, IN),
    "IN.7c.5+seg+pall": (V4 | {"seg", "pall"}, IN),
}


def main() -> None:
    rng = np.random.default_rng(config.RANDOM_SEED)
    s1_all = io_utils.load_source(config.TRAIN_SOURCE1)
    pools = {"S2": io_utils.load_source(config.TRAIN_SOURCE2), "S3": io_utils.load_source(config.TRAIN_SOURCE3)}
    all_true = io_utils.build_true_matches(io_utils.load_ground_truth())
    ck_idx = rng.choice(len(s1_all), size=50_000, replace=False)
    excluded = set(s1_all.iloc[ck_idx]["entity_id"])
    rest = s1_all[~s1_all["entity_id"].isin(excluded)]
    val = rest.iloc[np.random.default_rng(123).choice(len(rest), size=N_VAL, replace=False)].reset_index(drop=True)
    del rest
    with parallel_pool(DEFAULT_N_JOBS) as pool:
        vr = blocking.add_blocking_representations(val, DEFAULT_N_JOBS, pool=pool)
        prep = {k: blocking.add_blocking_representations(v, DEFAULT_N_JOBS, pool=pool) for k, v in pools.items()}
        if any("seg" in v for v, _ in VARIANTS.values()):  # vocabulary = the whole train S1, as in training
            costs = blocking_idf.build_segment_costs(
                blocking.add_blocking_representations(s1_all, DEFAULT_N_JOBS, pool=pool))
            for k, rep in prep.items():
                rep["name_seg"] = blocking_idf.segment_names(rep, costs)
                progress.log(f"segmented {int((rep['name_seg'] != '').sum())} glued {k} names")
    del s1_all
    del pools
    true_pairs = {(e, m) for e in vr["entity_id"] for m in all_true.get(e, ())}
    n_true = len(true_pairs)
    # slices keyed on the true pool record: empty address / native-script name
    pool_all = pd.concat(prep.values(), ignore_index=True).set_index("entity_id")
    empty_addr = set(pool_all.index[pool_all["addr_norm"] == ""])
    native = set(pool_all.index[~pool_all["name_norm"].map(str.isascii)])
    glued = set(pool_all.index[pool_all["name_norm"].map(
        lambda n: len(n.split()) == 1 and len(n) >= blocking_idf.SEG_MIN_LEN and n.isascii() and n.isalpha())])
    slices = {"emptyAddr": {p for p in true_pairs if p[1] in empty_addr},
              "native": {p for p in true_pairs if p[1] in native},
              "glued": {p for p in true_pairs if p[1] in glued}}
    del pool_all
    country = dict(zip(vr["entity_id"], vr["country"]))
    for name, (variant, fw) in VARIANTS.items():
        blocking_idf.record_tokens = _tokens(variant)
        c = pd.concat([blocking_idf.topk_candidates(vr, rep, k=10, label=f"-{src}", field_weights=fw)
                       for src, rep in prep.items()], ignore_index=True)
        pr = pipeline_v2.prune_candidates(c, 0.5, 3)
        for tag, d in (("K=10", c), ("pruned", pr)):
            hit = np.array([p in true_pairs for p in zip(d["source1_entity_id"], d["candidate_entity_id"])])
            by_c = {cty: hit[d["source1_entity_id"].map(country).to_numpy() == cty].sum()
                    / sum(1 for e, _ in true_pairs if country[e] == cty) for cty in ("India", "US")}
            got = set(zip(d["source1_entity_id"], d["candidate_entity_id"]))
            sl = " ".join(f"{k} {len(v & got) / len(v):.4f}" for k, v in slices.items())
            cty = d["source1_entity_id"].map(country)
            n_ent = pd.Series(country).value_counts()
            cpe = " ".join(f"{c} {(cty == c).sum() / n_ent[c]:.2f}" for c in ("India", "US"))
            progress.log(f"[{name:10s}] {tag:6s} recall {hit.sum() / n_true:.4f} "
                         f"(India {by_c['India']:.4f}, US {by_c['US']:.4f}) | {len(d) / N_VAL:.2f} cand/entity "
                         f"({cpe}) | {sl}")
    progress.log("DONE")


if __name__ == "__main__":
    main()
