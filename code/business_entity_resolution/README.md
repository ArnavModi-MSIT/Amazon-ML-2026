# Business Entity Resolution — pipeline (v9, submission #07, LB 0.971)

Approach write-up: [`../../APPROACH.md`](../../APPROACH.md).

## Setup

Python 3.10+, CPU only, ~26 GB RAM peak (the full 5M-record pools are held in memory during blocking).

```bash
pip install -r requirements.txt
```

The competition data goes in `dataset/train/` and `dataset/test/` at the repository root (not included);
outputs are written to `output/`.

## Run (from this directory)

```bash
# 1. Train + validate (~1 h): 150k train S1 entities blocked against the FULL train pools, fixed 20k held-out
#    validation set; learns the transliteration dictionary from train pairs (validation entities excluded)
python -u -m src.train_v2 --tag v9 --segment --field-weights '{"India": [0.7, 1.0, 0.5]}' \
    --country-floor '{"India": 0.55}' --n-train 150000 --latin-only-tokens --translit --key-channel --reverse

# 2. Test inference (~2.5 h): writes ../../output/matching_results.tsv and candidate_pairs.tsv, saves every
#    scored candidate to artifacts/test_scores_v9/, runs the official validator
python -u -m src.run_inference_v2 --tag v9
```

`artifacts/` already holds the trained `model_v9.txt`, its settings `thresholds_v9.json` (threshold, candidate
policy, text-representation version) and the learned `translit_v9.json`, so step 2 alone reproduces
submission #07. Inference refuses a model trained on a different text representation (`text_repr.REPR_VERSION`).

## Modules

| Module | Purpose |
|---|---|
| `src/config.py`, `src/io_utils.py` | paths, column names, TSV loading/writing |
| `src/text_repr.py` | normalization (Latin-diacritic fold, name/address cleanup, country-aware address tables), 9-script Indic transliteration skeleton, phonetic codes |
| `src/translit_dict.py` | word-level transliteration dictionary learned from train ground truth |
| `src/blocking.py` | per-record representations |
| `src/blocking_idf.py` | candidate generation: IDF token cosine top-K (field-weighted for India), glued-name segmentation, reverse top-1 pass |
| `src/pipeline_v2.py` | candidate channels (exact-key, reverse), pruning, S1-population / pool features, feature matrix |
| `src/features.py` | pairwise similarity features |
| `src/model.py` | LightGBM train / predict |
| `src/postprocess.py` | threshold decision + one-to-one conflict resolution |
| `src/evaluate.py` | local macro F0.5 scorer |
| `src/train_v2.py` | training + full-pool validation entry point |
| `src/run_inference_v2.py` | test inference entry point |
| `src/parallel_utils.py`, `src/progress.py` | process pool, progress/memory logging |
