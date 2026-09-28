"""Evaluate the Kaggle cross-encoder against / blended with v11 on the SAME test-like validation pairs.
Usage: python kaggle_nn/eval_nn_blend.py path/to/nn_val_scores.parquet   (run from student_resource/)"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution")
from src import io_utils  # noqa: E402

nn_path = sys.argv[1]
truth = io_utils.build_true_matches(io_utils.load_ground_truth())
v = pd.read_parquet("code/business_entity_resolution/reports/val_v11.parquet",
                    columns=["source1_entity_id", "candidate_entity_id", "label", "prob", "half"])
s = pd.read_parquet(nn_path, columns=["source1_entity_id", "candidate_entity_id", "nn"])
v = v.merge(s, on=["source1_entity_id", "candidate_entity_id"], how="left")
print(f"val pairs {len(v):,}; with NN score {v.nn.notna().mean():.2%}")
v["nn"] = v["nn"].fillna(0.0)
ents = pd.Index(v.source1_entity_id.unique()); e = ents.get_indexer(v.source1_entity_id); n = len(ents)
T = np.array([len(truth.get(x, ())) for x in ents], float); lab = v.label.to_numpy().astype(bool)
half = v.groupby("source1_entity_id").half.first().reindex(ents).to_numpy(); R, TU = half == "report", half == "tune"


def F(p, tau):
    emit = p >= tau
    tp = np.bincount(e, weights=emit & lab, minlength=n); k = np.bincount(e, weights=emit, minlength=n)
    return np.where((T == 0) & (k == 0), 1.0, np.where((T == 0) | (k == 0), 0.0, 1.25 * tp / (0.25 * T + np.maximum(k, 1))))


taus = np.round(np.arange(0.4, 0.96, 0.01), 2)
base = F(v.prob.to_numpy(), 0.80); rng = np.random.default_rng(0)


def ev(name, p):
    t = max(taus, key=lambda t: F(p, t)[TU].mean()); f = F(p, t); d = (f - base)[R]
    bs = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
    print(f"{name:<26} tau {t}: report {f[R].mean():.4f}  ({d.mean():+.4f}, 95% CI {np.percentile(bs, 2.5):+.4f}..{np.percentile(bs, 97.5):+.4f})")


print(f"v11 alone @0.80: report {base[R].mean():.4f}")
p11, pn = v.prob.to_numpy(), v.nn.to_numpy()
ev("NN alone", pn)
for w in (0.9, 0.8, 0.7, 0.6, 0.5):
    ev(f"{w:.1f}*v11 + {1 - w:.1f}*NN", w * p11 + (1 - w) * pn)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
best = None
for w in (0.9, 0.85, 0.8, 0.75, 0.7, 0.6):
    pb = 1 / (1 + np.exp(-(w * lg(p11) + (1 - w) * lg(pn))))
    t = max(taus, key=lambda t: F(pb, t)[TU].mean()); tune = F(pb, t)[TU].mean()
    if best is None or tune > best[0]: best = (tune, w, t)
    ev(f"logit {w:.2f}/{1 - w:.2f}", pb)
print(f"SELECTED on tune half: logit weight {best[1]}, threshold {best[2]}  -> pass these to make_blend_submission.py")
