"""Final decision stage: blend the LightGBM (v11) test scores with the cross-encoder scores and write
output/matching_results.tsv.

  p = sigmoid(0.8 * logit(p_v11) + 0.2 * logit(p_nn)); emit p >= 0.80 (France: ~0.893, the threshold that drops the same
  share of French pairs as v11's 0.80 -> 0.90, which scored higher on the leaderboard); then global one-to-one resolution
  (each Source 2/3 record keeps only its highest-probability Source 1 claimant, ties to the earlier Source 1 row).

Weights and threshold were tuned on the validation tune half and checked on the report half (+0.0011, 95% CI
+0.0005..+0.0018, replicated on two network runs); leaderboard 0.972398 -> 0.975. Pairs with p_v11 < 0.3 can never pass
this blend (checked on validation), so the network only scores pairs with p_v11 >= 0.3 (src/export_for_kaggle.py).
candidate_pairs.tsv (written by run_inference_v2) is unchanged: the same candidate set is scored by both models.

    python -m src.blend_nn --tag v11 --nn artifacts/nn_test_scores.parquet
"""
import argparse
import hashlib

import numpy as np
import pandas as pd

from . import config, io_utils

W, TAU, P_MIN = 0.8, 0.80, 0.3
FR_REF = (0.80, 0.90)  # France: drop the same share of French pairs as v11 did from 0.80 to 0.90 (leaderboard-tested)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v11")
    ap.add_argument("--nn", default=str(config.ARTIFACTS_DIR / "nn_test_scores.parquet"))
    args = ap.parse_args()

    s1 = pd.read_csv(config.TEST_SOURCE1, sep=config.SEP, usecols=["entity_id", "country"], dtype=str,
                     keep_default_na=False, na_values=[])
    s1_ids = s1["entity_id"].tolist()
    pos = pd.Series(np.arange(len(s1_ids)), index=s1["entity_id"].to_numpy())
    country = pd.Series(s1["country"].to_numpy(), index=s1["entity_id"].to_numpy())
    parts = [pd.read_parquet(f, filters=[("prob", ">=", P_MIN)])
             for f in sorted((config.ARTIFACTS_DIR / f"test_scores_{args.tag}").glob("batch_*.parquet"))]
    d = pd.concat(parts, ignore_index=True).rename(columns={"prob": "p_v11"})
    d = d.merge(pd.read_parquet(args.nn, columns=["source1_entity_id", "candidate_entity_id", "nn"]),
                on=["source1_entity_id", "candidate_entity_id"], how="left")
    print(f"pairs with p_v11 >= {P_MIN}: {len(d):,}; network coverage {d['nn'].notna().mean():.4%}")
    d["nn"] = d["nn"].fillna(d["p_v11"])
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    d["p"] = 1 / (1 + np.exp(-(W * lg(d["p_v11"].to_numpy()) + (1 - W) * lg(d["nn"].to_numpy()))))
    cty = country.reindex(d["source1_entity_id"].to_numpy()).to_numpy()
    d["pos"] = pos.reindex(d["source1_entity_id"].to_numpy()).to_numpy()
    fr = d[cty == "France"]
    ratio = (fr["p_v11"] >= FR_REF[1]).sum() / (fr["p_v11"] >= FR_REF[0]).sum()
    tau_fr = float(np.sort(fr["p"].to_numpy())[::-1][int(round((fr["p"] >= TAU).sum() * ratio)) - 1])
    print(f"France threshold {tau_fr:.6f} (keeps {ratio:.4f} of French pairs, as v11 0.80 -> 0.90)")
    kept = d[d["p"].to_numpy() >= np.where(cty == "France", tau_fr, TAU)]
    kept = kept.sort_values(["p", "pos"], ascending=[False, True], kind="stable")
    n0 = len(kept)
    kept = kept.drop_duplicates("candidate_entity_id", keep="first")
    mt = kept.groupby("source1_entity_id")["candidate_entity_id"].agg(lambda s: ",".join(sorted(s)))
    rows = pd.DataFrame({"source1_entity_id": s1_ids, "matched_entity_ids": mt.reindex(s1_ids).fillna("").to_numpy()})
    io_utils.write_id_list_file(rows, config.MATCHING_RESULTS_PATH, config.MATCHING_RESULTS_COLUMNS)
    print(f"{len(kept):,} matched pairs ({n0 - len(kept)} removed by one-to-one), "
          f"{(rows['matched_entity_ids'] != '').sum():,} entities matched | md5 "
          f"{hashlib.md5(config.MATCHING_RESULTS_PATH.read_bytes()).hexdigest()} -> {config.MATCHING_RESULTS_PATH}")


if __name__ == "__main__":
    main()
