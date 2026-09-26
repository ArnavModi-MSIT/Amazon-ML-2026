# Submission 04 — v3: IDF top-K blocking + relative-floor pruning + new features

**Score: 0.927** (leaderboard)

## What changed vs Submission 03 (0.519)

Real-scale diagnosis showed Submission 03's blocking only reached **0.326** of true pairs:
absolute frequency caps tuned on sampled pools masked almost every meaningful word at full
5M-record pool scale. The classifier already recovered ~99% of what blocking gave it, so the
entire loss was blocking.

- **Blocking:** IDF-weighted word-token cosine (name + transliteration-skeleton + address
  tokens, per country, per pool), top-10 per source, tokens with df > 50,000 dropped
  (`blocking_idf.py`). Recall ceiling 0.33 → 0.94.
- **Candidate pruning** (for the "smaller candidate set ranks higher" rule): keep the top 3 per
  entity/pool plus any candidate scoring ≥ 0.5 × that entity's best. 12.6 candidates/entity
  (vs 15.3 in Submission 03), recall ceiling ~0.924.
- **Model trained at real scale:** 54k train S1 entities blocked against the FULL train pools
  with the exact inference candidate policy (no sampled pools, no negative downsampling).
- **Features (49):** 35 pair string-similarity features (incl. new postcode match, first-name-
  token match), blocking score/rank/top/gap, pool frequency of the name/address keys, and
  within-entity relatives (score ÷ best, top-2 gap, count near the top, name/address similarity
  relative to the entity's best candidate).
- **Decision:** single threshold prob ≥ 0.65 (tuned on a separate half of validation), then
  global one-to-one conflict resolution.

## Validation (full-pool, held-out; threshold tuned on the other half)

macro F0.5 **0.9399** (India 0.925, US 0.950, singletons 0.924) at 12.3 candidates/entity.
On the same validation set, the Submission 03 pipeline scores 0.486 — its leaderboard score
was 0.519, so this validation has tracked the leaderboard. France (15% of test) has no labels.

## Output stats

- `matching_results.tsv`: 1,732,544 rows; 1,620,773 entities (93.5%) with ≥1 match;
  5,275,736 pairs (3.05/entity).
- `candidate_pairs.tsv`: 1,732,544 rows; 12.62 candidates/entity.
- Official validator: **PASS**. Runtime 3.0 h (blocking 55 min, 18 feature batches ~7.3 min each).

## Reproduce

From `code/business_entity_resolution/`:
```
python -u -m src.train_v2 --tag v3
python -u -m src.run_inference_v2 --tag v3
```
Config used: `thresholds_v3.json` (copied here).

## Files in this folder

GitHub rejects files over 100 MB, so the TSVs are stored gzipped (and
`candidate_pairs` is split in two). Restore them byte-for-byte with:
```
gzip -dc matching_results.tsv.gz > matching_results.tsv
cat candidate_pairs.tsv.gz.part00 candidate_pairs.tsv.gz.part01 | gzip -dc > candidate_pairs.tsv
```
MD5 of the restored files: matching_results.tsv `b18fa141f88183a1765a3306bed6bf96`,
candidate_pairs.tsv `b6c67a6bc556156fc7c8ae6b3dabd55a`.
