"""Re-decide test matches at other thresholds from the scores saved by run_inference_v2 (artifacts/test_scores_<tag>/),
without re-running blocking/featurization. Same decision path as run_inference_v2 (single threshold, then global
greedy one-to-one resolution keeping the highest-probability S1 claimant, ties to the earlier S1 row), so
`--tau` equal to the inference threshold reproduces its matching_results.tsv byte-for-byte. candidate_pairs.tsv is
unchanged (same scored set) and is not rewritten.

    python -m src.rethreshold --tag v9 --tau 0.80 0.85 0.88 --out ../../submissions/_rethreshold_v9
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import config, io_utils


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v9")
    ap.add_argument("--tau", type=float, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--country-tau", default=None, help='JSON {country: tau} overriding --tau for those countries')
    args = ap.parse_args()
    ctau = json.loads(args.country_tau) if args.country_tau else {}

    s1 = pd.read_csv(config.TEST_SOURCE1, sep=config.SEP, usecols=["entity_id", "country"], dtype="string",
                     keep_default_na=False, na_values=[])
    s1_ids = s1["entity_id"].tolist()
    pos = pd.Series(range(len(s1_ids)), index=s1["entity_id"].to_numpy())
    lo = min(list(args.tau) + list(ctau.values()))
    parts = []
    for f in sorted((config.ARTIFACTS_DIR / f"test_scores_{args.tag}").glob("batch_*.parquet")):
        parts.append(pd.read_parquet(f, filters=[("prob", ">=", lo)], dtype_backend="pyarrow"))  # lean: arrow strings
    sc = pd.concat(parts, ignore_index=True)
    sc["pos"] = pos.reindex(sc["source1_entity_id"].to_numpy()).to_numpy()
    sc = sc.sort_values(["prob", "pos"], ascending=[False, True], kind="stable")
    country = dict(zip(s1["entity_id"], s1["country"]))
    if ctau:
        sc["cty"] = pd.Series(sc["source1_entity_id"].to_numpy()).map(country).to_numpy()
    for tau in args.tau:
        thr = sc["cty"].map(ctau).fillna(tau).to_numpy() if ctau else tau
        kept = sc[sc["prob"].to_numpy() >= thr]
        n_before = len(kept)
        kept = kept.drop_duplicates("candidate_entity_id", keep="first")  # one-to-one: best claimant wins
        mt = kept.groupby("source1_entity_id")["candidate_entity_id"].agg(lambda s: ",".join(sorted(s)))
        rows = pd.DataFrame({"source1_entity_id": s1_ids,
                             "matched_entity_ids": mt.reindex(s1_ids).fillna("").to_numpy()})
        tag = f"tau_{tau:.2f}" + "".join(f"_{c}{t:.2f}" for c, t in sorted(ctau.items()))
        out = Path(args.out) / tag / "matching_results.tsv"
        io_utils.write_id_list_file(rows, out, config.MATCHING_RESULTS_COLUMNS)
        md5 = hashlib.md5(out.read_bytes()).hexdigest()
        empty = rows["matched_entity_ids"].eq("")
        by = pd.Series(empty.to_numpy(), index=[country[e] for e in s1_ids]).groupby(level=0).mean()
        print(f"tau {tau:.2f}: {len(kept)} pairs ({n_before - len(kept)} removed by one-to-one), "
              f"{len(kept) / len(s1_ids):.3f}/entity, empty {empty.mean():.2%} "
              f"({', '.join(f'{c} {v:.2%}' for c, v in by.items())}) | md5 {md5} -> {out}")


if __name__ == "__main__":
    main()
