# Amazon ML Challenge 2026 — Business Entity Resolution

Match each Source 1 business to its duplicates in Sources 2 and 3 (US, India, France). Scored on per-entity
macro F0.5, with smaller candidate sets ranked higher.

**Best submission: #07 (v9), leaderboard score 0.971** (held-out validation 0.9803, 13.3 candidates/entity).

| Path | What |
|---|---|
| [`APPROACH.md`](APPROACH.md) | Write-up: design, validation method, every version v1–v9 with measurements |
| [`code/business_entity_resolution/`](code/business_entity_resolution/) | Exactly the code used for submission #07, with the trained model, settings and learned dictionary |
| [`submissions/SCOREBOARD.md`](submissions/SCOREBOARD.md) | Every submission with local validation and real score |
| [`submissions/submission_07_v9_reverse/`](submissions/submission_07_v9_reverse/) | Submission #07 outputs (gzipped) + how they were generated |
| [`utils/validate_submission.py`](utils/validate_submission.py) | Official submission validator (called by the inference script) |

## Pipeline (v9) in one paragraph

Script-aware normalization (Latin-diacritic folding, record-id / honorific / look-alike-digit cleanup, country-aware
address canonicalization) with a transliteration dictionary learned from the training pairs plus a 9-script Indic
skeleton and phonetic codes; candidate generation by IDF-weighted token cosine (top-10 per source per entity, per
country, field-weighted for India, glued website names segmented into S1 words), plus an exact-key channel and a
reverse channel (each Source 2/3 record's single best Source 1 entity); relative-score pruning; 81 pairwise,
blocking, within-entity, S1-population and reverse features; LightGBM trained on 150k entities against the full
5M-record pools; one probability threshold (0.80) and global one-to-one conflict resolution. No external data.

## Reproduce

Place the competition data at `dataset/train/` and `dataset/test/`, then follow
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).
