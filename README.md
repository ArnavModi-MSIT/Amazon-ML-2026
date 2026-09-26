# Amazon ML Challenge 2026 — Business Entity Resolution

Match each Source 1 business to its duplicates in Sources 2 and 3 (US, India,
France). Scored on per-entity macro F0.5, with smaller candidate sets ranked
higher.

**Best submission so far: #04, leaderboard score 0.927** (held-out validation 0.940).

| Path | What |
|---|---|
| [`APPROACH.md`](APPROACH.md) | Full write-up of the approach, validation method and results |
| [`code/business_entity_resolution/`](code/business_entity_resolution/) | Pipeline code (the exact code that produced submission #04), trained `model_v3.txt` + thresholds |
| [`submissions/SCOREBOARD.md`](submissions/SCOREBOARD.md) | Every submission with local validation and real score |
| [`submissions/submission_04_v3_idf_topk_floor/`](submissions/submission_04_v3_idf_topk_floor/) | Submission #04 outputs (gzipped) + how they were generated |

## Pipeline (v3) in one paragraph

Unicode/script-aware normalization and a hand-built Indic→Latin transliteration
skeleton; candidate generation by IDF-weighted word-token cosine (top-10 per
source per entity, per country) with relative-score pruning (~12.6
candidates/entity); 49 pairwise, blocking and within-entity features; LightGBM
trained against the full 5M-record pools; single probability threshold (0.65)
and global one-to-one conflict resolution.

## Reproduce

The competition dataset is not included; place it at `dataset/train/` and
`dataset/test/` in the repo root (outputs go to `output/`). Then see
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md)
(`python -u -m src.train_v2 --tag v3`, then `python -u -m src.run_inference_v2 --tag v3`).
