# Amazon ML Challenge 2026 — Business Entity Resolution

Match each Source 1 business to its duplicates in Sources 2 and 3 (US, India,
France). Scored on per-entity macro F0.5, with smaller candidate sets ranked
higher.

**Best submission: #05 (v6), leaderboard score 0.962** (held-out validation 0.9712, 12.17 candidates/entity).

| Path | What |
|---|---|
| [`APPROACH.md`](APPROACH.md) | Full write-up: design, validation method, every version v1–v6 with measurements (§3.7–3.8) |
| [`code/business_entity_resolution/`](code/business_entity_resolution/) | The exact code that produced submission #05, with the trained `model_v6.txt` + `thresholds_v6.json` |
| [`submissions/SCOREBOARD.md`](submissions/SCOREBOARD.md) | Every submission with local validation and real score |
| [`submissions/submission_05_v6_segment_fieldweights/`](submissions/submission_05_v6_segment_fieldweights/) | Submission #05 outputs (gzipped) + how they were generated |

## Pipeline (v6) in one paragraph

Script-aware normalization (Latin-diacritic folding, look-alike-digit repair, address canonicalization incl.
country-specific French rules) with a hand-built transliteration skeleton for 9 Indic scripts and
phonetic consonant codes; candidate generation by IDF-weighted token cosine (top-10 per source per entity,
per country), with glued/website names segmented into S1-vocabulary words and field-weighted
name/address scoring for India; relative-score pruning (~12 candidates/entity); 71 pairwise, blocking,
within-entity and S1-population features; LightGBM trained on 150k entities against the full 5M-record
pools; single probability threshold (0.75) and global one-to-one conflict resolution.

## Reproduce

The competition dataset is not included; place it at `dataset/train/` and `dataset/test/` in the repo
root (outputs go to `output/`). Then see
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md):

```
python -u -m src.train_v2 --tag v6 --segment --field-weights '{"India": [0.7, 1.0, 0.5]}' --country-floor '{"India": 0.55}' --n-train 150000
python -u -m src.run_inference_v2 --tag v6
```
