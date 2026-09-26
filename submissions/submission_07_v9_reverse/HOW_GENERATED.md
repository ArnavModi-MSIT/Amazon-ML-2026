# Submission 07 — v9: v8 + reverse top-1 channel

**Score: 0.971** (0.970791, leaderboard, submitted 26 Sep 2026 22:38 IST)

## What changed vs Submission 06 (v8, 0.968)

- **Reverse top-1 channel** (idea from public repo mayankgoplani431-del; measured first on a Kaggle CPU notebook):
  every S2/S3 record finds its single best S1 record among the WHOLE S1 file of the split (same cosine as
  forward blocking). Those pairs are added as candidates (kept through pruning) and give three features to every
  candidate: `rev_top1` (this entity is the record's best S1), `rev_other` (another S1 is), `rev_best_ratio`
  (this pair's cosine / the record's best). Each S2/S3 record belongs to at most one S1 entity, so the record's
  own best S1 is a strong candidate even when the entity's forward top-K is crowded by look-alikes.
- Blocking recall ceiling (20k held-out validation entities, full pools): 0.9651 -> **0.9776** at 12.45 cand/entity.
- 81 features, single LightGBM (150k train entities), threshold 0.80.

## Validation (full-pool held-out report half; threshold tuned on the tune half)

Macro F0.5 **0.9803** (India 0.976, US 0.983, singletons 0.984). v8: 0.9768 (LB 0.968).

## Output stats (test)

- `matching_results.tsv`: 1,732,544 rows; 1,632,806 entities (94.2%) with >= 1 match; 5,741,233 pairs (3.31/entity).
- `candidate_pairs.tsv`: 1,732,544 rows; 13.29 candidates/entity (v8: 12.73; reverse channel +0.45).
- Per country: France 17.9 cand / 3.37 predicted / 5.3% empty / top-1 prob 0.954; India 13.5 / 3.26 / 6.0% / 0.947;
  US 11.2 / 3.36 / 5.7% / 0.950.
- One-to-one conflict resolution removed 0.08% of predicted pairs (v8: 0.22%).
- Official validator: **PASS**. Runtime 2.5 h.
- MD5: matching_results.tsv `3c05369c00c767f8d818e0de0a327ed5`, candidate_pairs.tsv `51fc6d25c29a8fca99a60eb67113416d`.
- Code: `code/business_entity_resolution_v9/` (frozen); `thresholds_v9.json`, `translit_v9.json` copied here.

## Reproduce

From `code/business_entity_resolution_v9/`:
```
python -u -m src.train_v2 --tag v9 --segment --field-weights '{"India": [0.7, 1.0, 0.5]}' --country-floor '{"India": 0.55}' --n-train 150000 --latin-only-tokens --translit --key-channel --reverse
python -u -m src.run_inference_v2 --tag v9
```

## Files in this folder (GitHub copy)

GitHub rejects files over 100 MB, so the TSVs are gzipped (`candidate_pairs` split in two). Restore byte-for-byte with:
```
gzip -dc matching_results.tsv.gz > matching_results.tsv
cat candidate_pairs.tsv.gz.part00 candidate_pairs.tsv.gz.part01 | gzip -dc > candidate_pairs.tsv
```
