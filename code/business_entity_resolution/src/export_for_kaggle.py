"""Export the test pairs the cross-encoder must score (p_v11 >= 0.3; lower pairs can never pass the blend in
src/blend_nn.py) with their "name | address" texts, built exactly like the training texts in kaggle_nn/.
Upload both files to the Kaggle notebook (kaggle_nn/ber_nn_all_in_one.py).

    python -m src.export_for_kaggle --tag v11 --out ../../kaggle_upload
"""
import argparse
from pathlib import Path

import pandas as pd

from . import config

P_MIN = 0.3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v11")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    parts = [pd.read_parquet(f, filters=[("prob", ">=", P_MIN)])
             for f in sorted((config.ARTIFACTS_DIR / f"test_scores_{args.tag}").glob("batch_*.parquet"))]
    pairs = pd.concat(parts, ignore_index=True).rename(columns={"prob": "p_v11"})
    ids = set(pairs["source1_entity_id"]) | set(pairs["candidate_entity_id"])
    texts = []
    for path in (config.TEST_SOURCE1, config.TEST_SOURCE2, config.TEST_SOURCE3):
        df = pd.read_csv(path, sep=config.SEP, dtype=str, keep_default_na=False, na_values=[], quoting=3)
        df = df[df["entity_id"].isin(ids)]
        texts.append(pd.DataFrame({"id": df["entity_id"],
                                   "text": df["business_name"].str.slice(0, 120) + " | " + df["business_address"].str.slice(0, 160)}))
    pairs.to_parquet(out / "test_pairs_v11.parquet", index=False, compression="zstd", compression_level=12)
    pd.concat(texts, ignore_index=True).to_parquet(out / "test_texts.parquet", index=False, compression="zstd",
                                                   compression_level=12)
    print(f"{len(pairs):,} pairs, {len(ids):,} records -> {out}")


if __name__ == "__main__":
    main()
