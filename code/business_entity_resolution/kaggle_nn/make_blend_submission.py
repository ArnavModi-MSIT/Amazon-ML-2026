"""Blend v11 (LightGBM) test scores with the Kaggle cross-encoder and write matching_results.tsv.
p = sigmoid(0.8 * logit(p_v11) + 0.2 * logit(p_nn)), threshold 0.77 (tuned on the validation tune half; report half
+0.0012, 95% CI +0.0005..+0.0019). Pairs with p_v11 < 0.3 are never emitted by this blend (checked on validation), so
only those >= 0.3 were scored by the network. France: stricter threshold that drops the same share of French pairs as
v11 0.80 -> 0.90 did (leaderboard 0.972174 -> 0.972398). Then global one-to-one resolution (best claimant wins, ties to
the earlier S1 row), exactly as in run_inference_v2. candidate_pairs.tsv is v11's, unchanged (same scored set).
Run from student_resource/:  python kaggle_nn/make_blend_submission.py nn_test_scores.parquet [W] [TAU]
"""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8      # weight of v11 in the logit blend (tuned on the tune half)
TAU = float(sys.argv[3]) if len(sys.argv) > 3 else 0.77   # blended threshold (tuned on the tune half)
FR_DROP_REF = (0.80, 0.90)
OUT = Path("submissions/submission_12_v11_nn_blend")

nn_scores = pd.read_parquet(sys.argv[1])
pairs = pd.read_parquet("kaggle_nn/test_pairs_v11.parquet")
s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"], dtype=str,
                 keep_default_na=False, na_values=[])
s1_ids = s1["entity_id"].tolist()
pos = pd.Series(np.arange(len(s1_ids)), index=s1["entity_id"].to_numpy())
country = pd.Series(s1["country"].to_numpy(), index=s1["entity_id"].to_numpy())

d = pairs.merge(nn_scores, on=["source1_entity_id", "candidate_entity_id"], how="left")
print(f"pairs {len(d):,}; NN coverage {d.nn.notna().mean():.4%}")
d["nn"] = d["nn"].fillna(d["p_v11"])
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
d["p"] = 1 / (1 + np.exp(-(W * lg(d["p_v11"].to_numpy()) + (1 - W) * lg(d["nn"].to_numpy()))))
d["cty"] = country.reindex(d["source1_entity_id"].to_numpy()).to_numpy()
d["pos"] = pos.reindex(d["source1_entity_id"].to_numpy()).to_numpy()

fr = d[d.cty == "France"]
ratio = (fr.p_v11 >= FR_DROP_REF[1]).sum() / (fr.p_v11 >= FR_DROP_REF[0]).sum()
target = int(round((fr.p >= TAU).sum() * ratio))
fr_tau = float(np.sort(fr.p.to_numpy())[::-1][target - 1])
print(f"France: v11 0.80->0.90 keeps {ratio:.4f} of French pairs -> blended France threshold {fr_tau:.4f}")


def write(tag, fr_threshold):
    thr = np.where(d.cty.to_numpy() == "France", fr_threshold, TAU)
    kept = d[d.p.to_numpy() >= thr].sort_values(["p", "pos"], ascending=[False, True], kind="stable")
    n0 = len(kept)
    kept = kept.drop_duplicates("candidate_entity_id", keep="first")
    mt = kept.groupby("source1_entity_id")["candidate_entity_id"].agg(lambda s: ",".join(sorted(s)))
    rows = pd.DataFrame({"source1_entity_id": s1_ids, "matched_entity_ids": mt.reindex(s1_ids).fillna("").to_numpy()})
    out = OUT / tag / "matching_results.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out, sep="\t", index=False)
    shutil.copy("submissions/submission_10_v11_testlike/candidate_pairs.tsv", out.parent / "candidate_pairs.tsv")
    empty = rows.matched_entity_ids.eq("")
    by = pd.Series(empty.to_numpy(), index=country.reindex(s1_ids).to_numpy()).groupby(level=0).mean()
    print(f"[{tag}] {len(kept):,} pairs ({n0 - len(kept)} removed by one-to-one), {len(kept) / len(s1_ids):.3f}/entity, "
          f"empty {empty.mean():.2%} ({', '.join(f'{c} {v:.2%}' for c, v in by.items())}) | "
          f"md5 {hashlib.md5(out.read_bytes()).hexdigest()}")
    r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(out), "--candidate",
                        str(out.parent / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("   validator:", r.stdout.strip().splitlines()[-1])


write("blend_France_strict", fr_tau)
write("blend_plain", TAU)
