# Business Entity Resolution — Pipeline (v6)

Approach write-up: [`../../../APPROACH.md`](../../../APPROACH.md).

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

## Run

All commands run from this directory (`code/business_entity_resolution/`).
Peak RAM is ~21 GB (full 5M-record pools are held in memory during blocking).

```bash
# 1. Train: 60k train S1 entities vs the FULL train pools; validates on a
#    fixed 20k held-out set (~30 min at 150k fit entities). v6 validation: macro F0.5 0.9712
python -u -m src.train_v2 --tag v6 --segment --field-weights '{"India": [0.7, 1.0, 0.5]}' \
    --country-floor '{"India": 0.55}' --n-train 150000

# 2. Inference on the test split (~3.5 h: blocking ~1 h + 18 feature batches);
#    writes ../../output/matching_results.tsv and ../../output/candidate_pairs.tsv,
#    saves every scored candidate to artifacts/test_scores_<tag>/, then runs the official validator
python -u -m src.run_inference_v2 --tag v6
```

Training writes `artifacts/model_<tag>.txt` and `artifacts/thresholds_<tag>.json`
(threshold, candidate policy, and the text-representation version); inference
reads both, so it reproduces exactly what was validated. Inference refuses a
model trained on a different text representation (`text_repr.REPR_VERSION`):
the v3 model behind submission #04 needs the v3 code (GitHub repo), not this one.

Optional: `python -m src.eda` (writes `reports/eda_report.md`).

## Module map

| Module | Purpose |
|---|---|
| `src/config.py` | paths, column names, constants |
| `src/io_utils.py` | load/write the `.tsv` files consistently |
| `src/text_repr.py` | Unicode/script-aware normalization, address canonicalization, 9-script transliteration skeleton, phonetic codes, legal-suffix stripping |
| `src/blocking.py` | builds per-record representations (`add_blocking_representations`, `add_representations`) |
| `src/blocking_idf.py` | candidate generation: IDF-weighted word/phonetic-token cosine, top-K per S1 entity |
| `src/features.py` | pairwise string-similarity features |
| `src/pipeline_v2.py` | candidate generation + feature matrix (adds blocking-score features) |
| `src/model.py` | LightGBM train/save/load/predict |
| `src/postprocess.py` | three-threshold per-entity decision rule + one-to-one conflict resolution |
| `src/evaluate.py` | local macro F0.5 scorer |
| `src/train_v2.py` | training + real-scale validation entry point |
| `src/run_inference_v2.py` | test inference entry point |
| `src/parallel_utils.py`, `src/progress.py` | process pool helpers, progress/memory logging |
| `src/eda.py` | data profiling |
| `src/_blocking_recall_analysis.py`, `src/_idf_topk_blocking_eval.py` | experiments that motivated the v2 blocking design |
| `src/_token_ablation.py` | v4 blocking-token ablation (recall ceiling per token variant) |
| `src/_cache_val_v2.py`, `src/_decision_rules_v2.py` | v3 decision-rule and pruning experiments |
| `src/_smoke_test_*.py` | quick checks for `text_repr` (with asserted v4 cases) and `features` |
