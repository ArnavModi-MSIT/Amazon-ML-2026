# Submission 05 — v6: glued-name segmentation + India field-weighted blocking + 71 features

**Score: 0.962** (leaderboard, submitted 26 Sep 2026 13:40 IST)

## What changed vs Submission 04 (v3, 0.927)

Full details: `APPROACH.md` §3.7 (v4) and §3.8 (v5/v6).

- **Text normalization (v4):** fake Latin accents folded, look-alike digits in names fixed (`M0dern`), `www`/`com`/
  `NULL` dropped, zero-padded house numbers stripped, street types / state names (incl. native script) / renamed
  cities canonicalized; transliteration for all 9 Indic scripts; phonetic (consonant-class) blocking tokens so a
  native-script name meets its English spelling. France-aware address table (departement ↔ region, `ST` = Saint).
- **Blocking (v5/v6):** glued/website names (`visioncarelynn.com`) segmented into words from the S1 file's own
  vocabulary; India scored as 0.7·cos(name) + 1.0·cos(address) with empty pool addresses counted as partial agreement;
  relative pruning floor 0.55 for India, 0.5 elsewhere. Validation recall ceiling 0.924 → 0.954.
- **Features (71):** S1-population name/address frequencies (whole S1 file of the split), name ownership by other S1
  businesses, invented-name share, pool distinct addresses per name / names per address, street-anchored house
  number, full-address frequency, legal-form family, name-missing flag; counts clipped for the train→test pool-size
  shift (US test pools are 0.62× train).
- **Bug fixes from code reviews:** digit Jaccard empty handling, LightGBM bagging actually enabled, `addr_key` used the
  house number twice, postcode leading zeros, unit numbers taken as house numbers.
- **Model:** LightGBM on 150k train S1 entities blocked against the full train pools (1.76M rows), best iteration
  3,168; single threshold 0.75; global one-to-one conflict resolution.

## Validation (full-pool, held-out; threshold tuned on the other half)

Report-half macro F0.5 **0.9712** (India 0.962, US 0.977, singletons 0.978) at 11.77 candidates/entity.
Previous: v3 0.9399 (LB 0.927), v4 0.9563, v5 0.9701. Validation has run ~0.013 above the leaderboard.

## Output stats (test)

- `matching_results.tsv`: 1,732,544 rows; 1,628,402 entities (94.0%) with ≥1 match; 5,580,925 pairs (3.22/entity).
- `candidate_pairs.tsv`: 1,732,544 rows; **12.17 candidates/entity** (Submission 04: 12.62).
- Per country: France 16.8 candidates / 3.22 predicted / 5.6% empty / mean top-1 prob 0.950; India 11.7 / 3.15 / 6.4% /
  0.943; US 10.9 / 3.32 / 5.7% / 0.948. France's prediction rate and confidence match India/US (no calibration shift);
  its candidate count is higher because French addresses share many common tokens.
- One-to-one conflict resolution removed only 0.17% of predicted pairs (9,780 / 5.59M) at full test scale.
- Official validator: **PASS**. Runtime 2.6 h (blocking 60 min, 18 feature batches ~5.4 min each).
- MD5: matching_results.tsv `f46c473bd19b68404787c881c40e066a`, candidate_pairs.tsv `cebeed3e0c4f259ebb5b13f2842279cc`.
- All scored candidates saved to `code/business_entity_resolution/artifacts/test_scores_v6/` (387 MB).

## Reproduce

From `code/business_entity_resolution/`:
```
python -u -m src.train_v2 --tag v6 --segment --field-weights '{"India": [0.7, 1.0, 0.5]}' --country-floor '{"India": 0.55}' --n-train 150000
python -u -m src.run_inference_v2 --tag v6
```
Config used: `thresholds_v6.json` (copied here).

## Files in this folder (GitHub copy)

GitHub rejects files over 100 MB, so the TSVs are stored gzipped (and `candidate_pairs` is split in two).
Restore them byte-for-byte with:
```
gzip -dc matching_results.tsv.gz > matching_results.tsv
cat candidate_pairs.tsv.gz.part00 candidate_pairs.tsv.gz.part01 | gzip -dc > candidate_pairs.tsv
```
