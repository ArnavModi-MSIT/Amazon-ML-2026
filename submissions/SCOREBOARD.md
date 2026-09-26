# Submission scoreboard

Tracks every submission's real (hidden-eval) score so we always know our top
scorer. Each submission gets its own folder here with the two output TSVs
plus a `HOW_GENERATED.md` describing exactly how it was produced.

| # | Folder | Method | Local val F0.5 | Real score | Notes |
|---|--------|--------|-----------------|------------|-------|
| 01 | [submission_01_fast_exact_key_only](submission_01_fast_exact_key_only/) | Exact-key blocking only, no TF-IDF, 50k-checkpoint model | 0.9398 | 0.433 | Smoke test for pipeline/format validation. Leaderboard top score at time of submission: ~0.987 -- large gap, expected: no fuzzy matching means most non-exact duplicates are missed. |
| 02 | [submission_02_splink_blocking](submission_02_splink_blocking/) | Splink (DuckDB) exact-key + word-overlap blocking, but model/thresholds trained on OLD TF-IDF-blocked candidates | 0.9398 (misleading -- see notes) | 0.518 | Confirmed a real train/inference blocking mismatch: local val score was measured against a different candidate distribution than real inference used, so it overstated real performance. 58.7% entity match rate. |
| 03 | [submission_03_splink_train_inference_consistent](submission_03_splink_train_inference_consistent/) | Splink 3-channel blocking (exact-key x2 + name word-overlap), model/thresholds retrained on the SAME blocking method | 0.7799 on sampled pools (misleading); **0.486 on full pools** | 0.519 | The sampled-pool validation overstated it; re-scored on full pools it gave 0.486, matching the real score. Root cause found later: blocking recall ceiling only 0.326 (absolute frequency caps masked most tokens at full scale). 15.3 candidates/entity. |
| 04 | [submission_04_v3_idf_topk_floor](submission_04_v3_idf_topk_floor/) | IDF word-token top-10 blocking + relative-floor pruning, 49 features (pool key frequency, within-entity relatives), LightGBM trained on full pools, threshold 0.65 | **0.9399** (full pools, held-out report half) | **0.927** | Recall ceiling 0.33 -> 0.92. 12.62 candidates/entity (smaller than #03). 93.5% entities matched, 3.05 pairs/entity. Validator PASS, 3.0 h runtime. Val (India/US only) 0.940 vs LB 0.927: consistent with France (15% of test, no labels) scoring below India/US. |
| 05 | [submission_05_v6_segment_fieldweights](submission_05_v6_segment_fieldweights/) | v6: normalization + 9-script transliteration + phonetic tokens (v4), glued-name segmentation + India field-weighted blocking, 71 features (S1-population frequencies, house number, address frequencies), review bug fixes, 150k training entities | **0.9712** (full pools, report half) | **0.962** | 12.17 candidates/entity (smaller than #04). 94.0% entities matched, 3.22 pairs/entity. France prediction rate/confidence match India/US. Conflict resolution removed 0.17%. Validator PASS, 2.6 h runtime. |
| 06 | [submission_06_v8_translit_keys](submission_06_v8_translit_keys/) | v8: v6 + learned transliteration dictionary (1,326 words from train pairs), name/address cleanup (record ids, honorifics, city typos, ordinals), native words dropped from blocking, exact-key channel, 78 features (ideas from public repos + teammate branch) | **0.9768** (full pools, report half) | **0.968** | 12.73 candidates/entity. 94.2% entities matched, 3.28 pairs/entity. Native-script blocking recall 0.87 -> 0.98. Validator PASS, 2.8 h runtime. |
| 07 | [submission_07_v9_reverse](submission_07_v9_reverse/) | v9: v8 + reverse top-1 channel (each S2/S3 record's best S1 among the whole S1 file, as candidates + 3 features) | **0.9803** (full pools, report half) | **0.971** (0.970791) | 13.29 candidates/entity. Recall ceiling 0.965 -> 0.978. Conflicts removed 0.08%. Validator PASS, 2.5 h. |

**Current top scorer: #07 (0.971)**.

Note: a v2 submission (IDF top-K, no pruning, val 0.934) was also uploaded between #03 and #04 but did not succeed, so it has no score and no folder here.

Fallbacks with frozen code: v5 (val 0.9701) in `code/business_entity_resolution_v5/`, v4 (0.9563) in `_v4/`.
Submission #04's code is in the GitHub repo (the v3 model is archived; current code refuses it by design). Full-pool local validation has tracked the leaderboard every time (0.486 vs 0.519; 0.940 vs 0.927; 0.971 vs 0.962; 0.977 vs 0.968; 0.980 vs 0.971).
