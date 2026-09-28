# Business Entity Resolution: final pipeline (LightGBM v11 + cross-encoder blend, leaderboard 0.976331)

Methodology: [`../../SOLUTION.md`](../../SOLUTION.md).

```
test records -> candidate generation (IDF top-K + exact-key + reverse channels, pruning)   [src/run_inference_v2.py]
             -> 87 features -> LightGBM v11 probability for every candidate                  [saved: artifacts/test_scores_v11/]
             -> cross-encoder probability for candidates with p_v11 >= 0.3 (Kaggle GPU)      [kaggle_nn/ber_nn_all_in_one.py]
             -> blend + threshold + one-to-one resolution -> output/matching_results.tsv      [src/blend_nn.py]
output/candidate_pairs.tsv = every candidate the models score (written by run_inference_v2; 13.29 per Source 1 entity)
```

## Setup

Python 3.10.11 on Windows 11, CPU only, 32 GB RAM (peak ~26 GB during candidate generation).
The cross-encoder step runs on a Kaggle notebook (GPU T4 x2, Internet on) with PyTorch + transformers preinstalled.

```bash
pip install -r requirements.txt
```

Put the competition data in `dataset/train/` and `dataset/test/` next to `code/` (the `student_resource` layout).
Outputs go to `output/` at that level.

## Fast path: regenerate the submitted files (no GPU needed)

`artifacts/` contains the trained LightGBM `model_v11.txt` (+ `thresholds_v11.json`, `translit_v11.json`) and the
cross-encoder's test scores `nn_test_scores.parquet`.

```bash
# 1. LightGBM inference (~2.5 h): writes output/candidate_pairs.tsv, output/matching_results.tsv (v11 alone) and
#    saves every scored candidate to artifacts/test_scores_v11/
python -u -m src.run_inference_v2 --tag v11
# 2. Final decision: blend with the cross-encoder scores -> output/matching_results.tsv (md5 935c7f5c464d0066c7e0d40a868047c5)
python -m src.blend_nn --tag v11 --nn artifacts/nn_test_scores.parquet
```

## Full retraining from the raw data

```bash
# a. LightGBM v11 (~50 min). --universe-frac 0.81 keeps 81% of the train S1 file as the S1 universe (reverse channel,
#    S1-population counts) to match the test's share of ownerless pool records (~40% vs ~26% in train).
python -u -m src.train_v2 --tag v11 --segment --field-weights '{"India": [0.7, 1.0, 0.5]}' \
    --country-floor '{"India": 0.55}' --n-train 150000 --latin-only-tokens --translit --key-channel --reverse \
    --lgb-params '{"num_leaves": 127, "max_depth": -1, "min_child_samples": 100, "colsample_bytree": 0.7}' \
    --universe-frac 0.81
# b. Cross-encoder training pairs (~50 min): same run with the meta-blocking filter score and a pair dump
#    -> reports/pairs_v16.parquet (ids, label, role fit/es/tune/report, filter score p1; no text)
python -u -m src.train_v2 --tag v16 --segment --field-weights '{"India": [0.7, 1.0, 0.5]}' \
    --country-floor '{"India": 0.55}' --n-train 150000 --latin-only-tokens --translit --key-channel --reverse \
    --lgb-params '{"num_leaves": 127, "max_depth": -1, "min_child_samples": 100, "colsample_bytree": 0.7}' \
    --universe-frac 0.81 --stage1-recall 0.9998 --dump-pairs
# c. LightGBM test inference (~2.5 h), as in the fast path
python -u -m src.run_inference_v2 --tag v11
# d. Export the test pairs to score (p_v11 >= 0.3) with their texts
python -m src.export_for_kaggle --tag v11 --out ../../kaggle_upload
# e. On Kaggle (GPU T4 x2, Internet on): attach train_source1/2/3.tsv, reports/pairs_v16.parquet and the two files
#    from step d as datasets FIRST (attaching later restarts the session), then run kaggle_nn/ber_nn_all_in_one.py in
#    one cell (~55 min): fine-tunes sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M)
#    as a pair classifier on 600k training pairs, writes nn_val_scores.parquet and nn_test_scores.parquet
# f. (optional) check the blend on validation: python kaggle_nn/eval_nn_blend.py nn_val_scores.parquet  (from student_resource/)
# g. Final decision
python -m src.blend_nn --tag v11 --nn path/to/nn_test_scores.parquet
```

## Modules (`src/`)

| Module | Purpose |
|---|---|
| `config.py`, `io_utils.py` | paths, column names, TSV loading and writing |
| `text_repr.py` | normalization (Latin-diacritic fold, name/address cleanup, country-aware address tables), 9-script Indic transliteration skeleton, phonetic codes |
| `translit_dict.py` | word-level transliteration dictionary learned from train ground truth |
| `blocking.py` | per-record blocking representations |
| `blocking_idf.py` | candidate generation: IDF token cosine top-K (field-weighted for India), glued-name segmentation, reverse top-1 pass, optional empty-address channel (off) |
| `pipeline_v2.py` | candidate channels (exact-key, reverse), pruning, S1-population and pool counts, feature matrix |
| `meta_blocking.py` | optional supervised meta-blocking filter (off in v11; used to dump the cross-encoder's training pairs) |
| `features.py` | pairwise name/address similarity features |
| `model.py`, `postprocess.py` | LightGBM training/prediction; threshold + one-to-one resolution |
| `evaluate.py` | local macro F0.5 scorer |
| `train_v2.py` | training + full-pool validation (`--compare-models` scores earlier models on the same rows) |
| `run_inference_v2.py` | LightGBM test inference (candidates, scores, validator) |
| `export_for_kaggle.py`, `blend_nn.py` | cross-encoder input export; final blend and decision |
| `rethreshold.py` | re-decide LightGBM matches at other thresholds from saved scores |
| `kaggle_nn/` | Kaggle cross-encoder notebook script and the validation blend evaluation |
