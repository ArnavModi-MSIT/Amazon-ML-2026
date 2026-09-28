# Amazon ML Challenge 2026 — Business Entity Resolution

An end-to-end entity-resolution system for matching each Source 1 business to duplicate records in Sources 2 and 3 across the United States, India, and France.

## Result

**Best leaderboard score: 0.976331 (submission #13).**

The final system combines a LightGBM candidate classifier with a fine-tuned multilingual MiniLM cross-encoder. It achieves a test-like held-out macro F0.5 score of **0.9844** while producing an average of **13.29 candidates per Source 1 entity**.

## How it works

1. Normalize multilingual business names, addresses, phone numbers, and websites.
2. Generate a small candidate set using IDF-weighted token similarity, exact keys, and a reverse best-match channel.
3. Score 87 pairwise and candidate-context features with LightGBM.
4. Re-score difficult candidates with a multilingual cross-encoder trained on challenge data.
5. Blend both model probabilities, apply country-aware thresholds, and resolve matches globally with a one-to-one constraint.

Important design choices include support for nine Indic scripts, a transliteration dictionary learned only from training pairs, glued-name segmentation, test-density-matched training, and full-pool validation instead of artificially easy sampled pools.

## Repository structure

| Path | Contents |
|---|---|
| [`SOLUTION.md`](SOLUTION.md) | Detailed methodology, experiments, validation, and results |
| [`APPROACH.md`](APPROACH.md) | Development history and measured model iterations |
| [`code/business_entity_resolution/`](code/business_entity_resolution/) | Final inference/training pipeline, model artifacts, and neural scoring code |
| [`submissions/SCOREBOARD.md`](submissions/SCOREBOARD.md) | Complete submission history with local and leaderboard scores |
| [`submissions/submission_13_nn2/`](submissions/submission_13_nn2/) | Best submission outputs and generation notes |
| [`utils/validate_submission.py`](utils/validate_submission.py) | Official-format submission validator |

## Reproduce the final output

### Requirements

- Python 3.10+
- About 32 GB RAM for full inference
- A CUDA GPU or Kaggle notebook for neural re-scoring
- Competition data placed under `dataset/train/` and `dataset/test/`

Install the dependencies:

```bash
cd code/business_entity_resolution
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run LightGBM inference, then blend the included neural scores:

```bash
python -u -m src.run_inference_v2 --tag v11
python -m src.blend_nn --tag v11 --nn artifacts/nn_test_scores.parquet
```

The expected best-submission checksum is documented in [`submissions/submission_13_nn2/HOW_GENERATED.md`](submissions/submission_13_nn2/HOW_GENERATED.md). For full retraining and GPU steps, see the [pipeline README](code/business_entity_resolution/README.md).

## Data and licensing note

The competition dataset is not redistributed. The repository contains the code, trained tabular model, learned challenge-data artifacts, neural prediction scores, and compressed submission outputs required to inspect and reproduce the final decision stage. Verify the competition rules and the pretrained model license before redistributing artifacts or using the system commercially.
