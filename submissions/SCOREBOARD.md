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

**Current top scorer: #04 (0.927)**. Full-pool local validation has tracked the leaderboard both times (0.486 vs 0.519; 0.940 vs 0.927).
