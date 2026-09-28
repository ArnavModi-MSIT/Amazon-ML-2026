# Amazon ML Challenge 2026: Business Entity Resolution Solution

**Final submission:** #13

**Submission date:** 27 September 2026

**Leaderboard score:** 0.976331

---

## 1. Executive Summary

A three-stage pipeline: multi-channel candidate generation, a LightGBM pair classifier, and a small multilingual
cross-encoder (a fine-tuned transformer that reads both records together) whose probability is blended with
LightGBM's, followed by one threshold and global one-to-one resolution. Blocking is IDF-weighted word-token cosine computed per country and per
source, plus an exact name+number key channel and a reverse channel that gives each Source 2/3 record's single best
Source 1 entity. The text side is script-aware (9 Indic scripts, a transliteration dictionary learned from the
training pairs, glued-name segmentation, country-aware address canonicalization). There are 87 pair, blocking,
candidate-list and Source-1-population features.

Two things mattered most:
- Every decision was validated on held-out entities blocked against the **full** 5M-record pools. Sampled-pool
  validation was off by 0.26 F0.5.
- The final model is trained with the **test's density of ownerless records**: 40% of test pool records belong to no
  Source 1 entity, against 26% in train.

Held-out macro F0.5 is **0.9844** under test-like conditions (LightGBM alone: 0.9820). The public leaderboard score
is **0.976331** (LightGBM alone: 0.972174). The candidate set averages 13.29 records per Source 1 entity.

A third finding decided the final submission. The test pools contain about **1.7× more "sibling decoys"** per entity
than train: records with the same name and street but a house number moved a few doors, or a sister company's name
at the same address. The cross-encoder separates these look-alikes from true copies better than string features do.
The first cross-encoder improved the leaderboard by 0.0026; training the final network on all 1.32M pairs for two epochs improved it further.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the training data:
- **Country is a hard partition.** Among 7.64M true pairs there are zero cross-country matches, so every step runs
  per country.
- **Match counts.** On average there are 3.46 true matches per Source 1 entity (1.67 in S2, 1.79 in S3). 5.6% of
  entities are singletons, which score 1.0 only with an empty prediction.
- **Distractors are real businesses.** In train, 73–75% of S2/S3 records are a true match of some Source 1 entity, so
  most distractors are genuine records of *other* businesses: chain branches, same-name businesses, and sister
  companies at the same address. A randomly sampled pool has far fewer look-alikes. Validating on sampled pools
  predicted 0.78 for a model that scored 0.519 on the leaderboard. Re-scored against the full pools, it gave 0.486.
- **Test differs from train.** The test file has 1.73M Source 1 entities against 9.97M pool records. Train has 2.21M
  against 10.32M. So about 40% of test pool records have no owner in the Source 1 file, compared with 26% in train.
  France (15% of test) has no training labels at all.
- **Noise patterns:**
  - typos and character corruption (`consu1tants`, `Çlub`, fake Latin accents that Source 1 never has)
  - word reordering and legal-suffix variation (`SARL`, `Pvt Ltd` / `Private Limited`, `LLC`)
  - glued website-style names (`visioncarelynn.com`, `southgujaratfoodscom`)
  - Indic-script names in S2/S3 (24% of India S2 names; Source 1 is 100% Latin)
  - **empty addresses** in S2/S3
  - zero-padded or look-alike house numbers (`0023`)
  - state and département variants (`Tamil Nadu` / `TN` / `தமிழ்நாடு`; Nord vs Hauts-de-France)
  - invented replacement names at a real address (`Novivera`, `Belowex`)

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier + neural re-scorer (candidate generation → LightGBM → multilingual
cross-encoder blend → threshold + one-to-one constraint)
**Core Innovation:** four ideas, all using only the challenge data (the one pretrained model is Apache-2.0, 118M parameters):
- **Full-scale, distribution-matched training and validation.** The model trains on the real candidate distribution
  against the full pools, and on the final model the Source 1 "universe" is thinned to the test's ownerless-record
  density.
- **A reverse candidate channel and its features.** Each pool record contributes its best Source 1 entity across the
  whole Source 1 file. "Another Source 1 entity is this record's best match" became the model's strongest feature.
- **Script-aware normalization.** A transliteration dictionary learned from the training pairs, with validation
  entities excluded.
- **A neural re-scorer where it matters.** A small multilingual cross-encoder, fine-tuned on our own training pairs,
  is blended with LightGBM. It reads Indic scripts and French natively, and it is best at telling a true copy from a
  sister-company or neighbouring-house decoy, which the test set has 1.7× more of than train.

---

## 3. Candidate Generation (Blocking)

All steps run separately per country and per pool (S2, S3):

- **IDF top-K cosine.**
  - Each record becomes a bag of tokens: normalized name words and their transliteration skeletons (`n:`), address
    words (`a:`), and phonetic codes (`p:`) for Source 1 and non-ASCII pool names.
  - TF-IDF (sublinear TF) is fit on the pool. Tokens with document frequency above 50,000 are dropped.
  - Each Source 1 record keeps its **top 10** pool records by cosine (`sparse_dot_topn`).
  - India uses field-weighted scoring, 0.7·cos(name) + 1.0·cos(address). An empty pool address counts as partial
    agreement (0.5 × the name cosine), so empty-address duplicates are not outranked by same-name decoys.
  - Pool names that are one long glued token are segmented into Source 1 vocabulary words by dynamic programming.
- **Exact-key channel.** Pairs sharing the suffix-free name core plus house number (or a rare exact name) are always
  candidates (at most 5 per key).
- **Reverse top-1 channel.** For every pool record, the Source 1 record with the highest cosine across the **whole**
  Source 1 file is added as a candidate. Each pool record belongs to at most one entity, so its best entity is a
  natural candidate even when that entity's own top-10 is crowded by look-alikes.
- **Pruning.** A candidate is kept if its rank is below 3, or its score is at least 0.5 × the entity's best score in
  that pool (0.55 for India). Exact-key and reverse-top-1 pairs are always kept.
- *Measured and not used:* an empty-address channel (each Source 1 record's top-2 among empty-address pool records,
  targeting the 48% of blocking misses with an empty pool address; `--empty-k`, off by default) raised the recall
  ceiling 0.9781 → 0.9791 but cost +9% candidates and −0.0003 F0.5, so it is disabled.

- **Blocking keys used:** IDF-weighted name, skeleton, phonetic and address tokens; exact name-core + house-number
  keys; reverse best-match.
- **Candidate pairs generated:** 23,022,401 for 1,732,544 Source 1 entities (13.29 per entity; 0 entities
  without candidates).
- **How we ensured true matches were not lost:**
  - Every blocking change was accepted only if it raised the **pruned recall ceiling** on 20k held-out entities
    blocked against the full train pools.
  - The recall ceiling rose from 0.33 (key blocking with frequency caps) to 0.92 (IDF top-K), then to **0.978**
    (segmentation, field weights, transliteration, exact keys, reverse channel), at about 13 candidates per entity.
  - The complementary channels recover specific miss types: glued names, native scripts, crowded look-alikes, exact
    keys.

---

## 4. Matching Model

**Features used (87):**
- **Name features.** Char 3-gram and token Jaccard (raw and transliteration skeleton), token containment,
  Levenshtein, token-sort, partial ratio, Jaro-Winkler, LCS, phonetic-code Jaccard, and joined-name similarity. Also:
  legal-suffix match and family (Ltd = Limited), script match, acronym match, and name-number conflict. IDF-weighted
  name overlap covers the weight of shared words and the IDF of the rarest word present on only one side, so
  "Alizes Primaire" vs "Alizes Gestion" counts as evidence against a match.
- **Address features.**
  - Char n-gram and token Jaccard, Levenshtein, token-sort, digit-token Jaccard (−1 when a side has no digits), and
    missing-address flags.
  - A street-anchored house number: match, edit distance, and log difference.
  - Number+street key match and IDF-weighted address overlap.
- **Other:**
  - Blocking score, rank, top score and gap; exact-key flag.
  - Reverse-channel flags: is this Source 1 entity the pool record's best match (`rev_top1`), is another entity its
    best (`rev_other`), and the ratio to that best score.
  - Within-entity relatives (candidates within 90% of the best score, same-name/same-address counts among the
    entity's candidates).
  - Source-1-population counts over the whole Source 1 file: how many Source 1 businesses share the candidate's name
    core, own that exact name, or sit at its address, and the share of its name words that appear anywhere in Source 1.
  - Pool counts: distinct addresses per name, distinct names per address. All counts are clipped at 8 (pool) or
    4 (Source 1), because test pools are smaller than train pools.
  - Target source.

**Model type:** LightGBM binary classifier.
- **Settings:** 127 leaves, no depth limit, min_child_samples 100, feature fraction 0.7, bagging 0.8, learning rate
  0.05, early stopping on 6k separate entities.
- **Training data:** 150k training Source 1 entities, each blocked against the **full** train pools with the exact
  inference pipeline. All candidates are kept (~2M rows, 1:3 positive:negative), with no downsampling, so
  probabilities stay calibrated.
- **Test-like density:** the Source 1 universe (used by the reverse channel and the population counts) keeps 81% of
  train Source 1 entities, which reproduces the test's 40% ownerless pool records. On identical test-like validation
  rows this model beats the same model trained on the full universe by +0.0013 (95% CI +0.0006 to +0.0020).
- **Most important features (gain):** `rev_other`, `rev_top1`, address digit-token Jaccard, house-number edit
  distance, skeleton token-sort ratio.

**Threshold selection method:**
- One probability threshold, tuned for macro F0.5 on the tuning half of the held-out entities and reported on the
  untouched other half. The result is 0.80.
- Three-threshold singleton gates, per-country, per-source and rank-aware thresholds, score-gap and margin rules,
  per-source caps, top-1 rescue, and a second-stage listwise model were all measured. None beat the single threshold.
- Leaderboard checks on the previous model agreed: 0.75 → 0.970469, 0.80 → 0.970791, 0.88 → 0.970398.
- After thresholding, global greedy one-to-one resolution keeps only the highest-probability Source 1 claimant for
  each pool record (verified on train: no S2/S3 record ever matches two Source 1 entities).

**Neural re-scorer (final stage):**
- **Model.** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M parameters, multilingual),
  fine-tuned for one epoch as a pair classifier on "name | address" of both records. Trained on 600k training pairs
  drawn from the same candidate distribution LightGBM sees (Kaggle, 2× T4 GPU, ~15 min). It is weaker than LightGBM
  on its own (0.959 vs 0.982) but makes different mistakes.
- **Blend.** p = sigmoid(0.8·logit(p_LightGBM) + 0.2·logit(p_network)), threshold 0.80. Weights and threshold were
  chosen on the validation tune half. On the report half the blend gains **+0.0011 (95% CI +0.0005 to +0.0018)**,
  replicated on two independent network runs.
- **Cost.** Candidates with LightGBM probability below 0.3 can never pass the blend (checked on validation), so the
  network only scores the 6.0M test pairs above it (3.5 per entity), in ~40 minutes on the GPU.
- **France.** No labels exist, so its threshold is set from leaderboard evidence. Making French predictions stricter
  (LightGBM 0.80 → 0.90, dropping 1.8% of French pairs) raised the leaderboard from 0.972174 to 0.972398, so the
  blend drops the same share of French pairs (threshold 0.893).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - **0.9844** under test-like ownerless density with the final two-epoch network (LightGBM alone 0.9820).
  - Under the standard density, the previous model v10 scored 0.9826 (India 0.9799, US 0.9845).
  - The public leaderboard score is **0.976331** (LightGBM alone 0.972174; + France threshold 0.972398).
  - Full-pool validation tracked the leaderboard throughout: 0.486→0.519, 0.940→0.927, 0.971→0.962, 0.977→0.968,
    0.980→0.971.

| Version | Main change | Held-out F0.5 | Leaderboard |
|---|---|---|---|
| v1 | key blocking with frequency caps | 0.486 | 0.519 |
| v3 | IDF top-K blocking + relative-floor pruning | 0.940 | 0.927 |
| v6 | transliteration, segmentation, India field weights, Source-1-population features | 0.971 | 0.962 |
| v8 | learned transliteration dictionary, cleanup, exact-key channel | 0.977 | 0.968 |
| v9 | reverse top-1 channel + features | 0.980 | 0.971 |
| v10 | IDF-overlap / acronym / number-conflict features, 127-leaf trees | 0.983 | — |
| v11 | v10 trained at test-like ownerless density (`--universe-frac 0.81`) | 0.9820 (test-like; v10 0.9807, v9 0.9786 on the same rows) | 0.972174 |
| v11 + France 0.90 | stricter threshold for France only (no labels; leaderboard-tested) | — | 0.972398 |
| v11 + initial cross-encoder | first multilingual MiniLM blend | 0.9831 (test-like) | 0.975 |
| **final (#13)** | v11 + MiniLM trained on all 1.32M pairs for two epochs | **0.9844** (test-like) | **0.976331** |

The remaining loss on validation splits into three parts (v10, report half, total 0.0175):
- blocking misses: 0.0065
- true matches in the candidates but scored below threshold: 0.0085
- false positives: 0.0026

- **Common false positives (wrong merges):** only 0.27% of predicted pairs on validation are wrong. They are almost
  all:
  - **sister companies at the same address**, sharing a distinctive word: `MRF Digital` / `MR Digital`,
    `Houston Metro Railway` / `Houston Railway Partners`, `Alizes Primaire` / `Alizes Gestion`
  - **invented names at a known address**: `Belowex`, `Slneo`
  - near-identical names with a different house number (`8748` vs `874 Mingo Rd`)
- **Common false negatives (missed matches):**
  - Blocking misses (2.3% of true pairs): 48% are pool records with an **empty address** whose name is also noisy
    (`Eye Praales Group`, `Dynamic Transit LLC Center`). 17% are single-token or website-style names
    (`econsultingcom`); 9% are native-script names.
  - Rejected candidates are mostly records whose name was **replaced by an unrelated brand-like word** at the
    correct address (`Novivera`, `Orbiiri`). They are indistinguishable from the invented-name decoys above, and the
    F0.5 metric correctly declines them.

---

## 6. Conclusion

- Most of the score came from making candidate generation and validation **faithful to the real scale and
  distribution**. Moving to full pools and IDF top-K raised the recall ceiling from 0.33 to 0.92. Complementary
  channels (reverse best-match, exact keys, segmentation, transliteration) took it to 0.978. Training at the test's
  ownerless-record density added more.
- The final and largest single gain came from a small neural re-scorer. On the leaderboard it was worth 2.4× what
  validation predicted, because the test set has more sibling decoys than training, and those are exactly the cases
  where reading both records together beats hand-built string features.
- The decision layer turned out to be saturated. Probabilities are well calibrated, and every rule more complex
  than one threshold with one-to-one resolution failed on held-out data.
- **Lesson:** measure every change on the real distribution, paired, before believing it.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/` holds the source code, and `kaggle_nn/` holds the GPU notebook script for the cross-encoder.
- `README.md` has the exact commands for both the fast path and full retraining.
- `requirements.txt` pins the dependencies.
- `artifacts/` holds the trained LightGBM `model_v11.txt`, its settings `thresholds_v11.json`, the learned dictionary
  `translit_v11.json`, and the cross-encoder's test scores `nn_test_scores.parquet`.

Fast path, reproducing the submitted files without a GPU (run from that folder, data in `dataset/train` and `dataset/test`):

```bash
# LightGBM inference -> output/candidate_pairs.tsv and saved scores for every candidate (~2.5 h)
python -u -m src.run_inference_v2 --tag v11
# blend with the cross-encoder scores -> output/matching_results.tsv (md5 935c7f5c464d0066c7e0d40a868047c5)
python -m src.blend_nn --tag v11 --nn artifacts/nn_test_scores.parquet
```

Full retraining (LightGBM training, the cross-encoder's training-pair dump, pair export, and the Kaggle notebook
run) is listed step by step in the README.

| Module | Purpose |
|---|---|
| `text_repr.py`, `translit_dict.py` | normalization, Indic transliteration skeleton, phonetic codes, learned dictionary |
| `blocking.py`, `blocking_idf.py` | record representations; IDF top-K, segmentation, reverse pass, optional empty-address channel |
| `pipeline_v2.py`, `meta_blocking.py` | candidate channels, pruning, feature matrix (87 features); optional meta-blocking filter |
| `features.py` | pairwise similarity features |
| `model.py`, `postprocess.py` | LightGBM; threshold + one-to-one resolution |
| `train_v2.py`, `run_inference_v2.py` | training/validation and LightGBM inference entry points |
| `export_for_kaggle.py`, `kaggle_nn/ber_nn_all_in_one.py` | cross-encoder input export; fine-tuning and scoring on a Kaggle GPU |
| `blend_nn.py` | final blend, France threshold, one-to-one resolution -> `matching_results.tsv` |
| `rethreshold.py` | re-decide LightGBM matches at another threshold from saved test scores (no re-run) |

### B. Additional Results

- **Calibration on the tuning half.** Predicted probability against the actual match rate: 0.75–0.80 → 0.74,
  0.80–0.85 → 0.87, 0.85–0.90 → 0.93, 0.95–0.99 → 0.99. Only about 1% of pairs fall between 0.5 and 0.95.
- **France (no labels).** On test, France looks like India and the US: 5.2% empty predictions (India 6.0%, US 5.7%)
  and 94% of entities with a top probability of at least 0.95. A manual audit of 25 random French matches found
  mostly correct merges; the few doubtful ones were sister companies at the same address.
- **Runtime.** On one CPU-only laptop (32 GB RAM, 16 threads), training plus validation takes about 50 min and
  LightGBM test inference 2 h 22 min (peak ~25 GB). The cross-encoder takes ~55 min on a Kaggle 2× T4 GPU notebook, and
  the final blend ~3 min on the laptop.
