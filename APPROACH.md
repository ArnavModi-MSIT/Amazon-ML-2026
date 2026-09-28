# Amazon ML Challenge 2026 — Business Entity Resolution: Approach (final model)

_Final status as of 27 Sep 2026._

**Version history (full-pool held-out validation → leaderboard):**

| Version | What changed | Val macro F0.5 | Leaderboard |
|---|---|---|---|
| v1 | key/token blocking with frequency caps | 0.486 | 0.519 (#03) |
| v2 | IDF word-token top-K blocking (§3.2) | 0.934 | submitted, no score (did not succeed) |
| v3 | + relative-floor pruning, pool-frequency + within-entity features | 0.9399 | **0.927** (#04) |
| v4 | + normalization fixes, 9-script transliteration, phonetic blocking tokens, 3 features | 0.9563 | not submitted |
| v5 | + glued-name segmentation, India field-weighted blocking, S1-population / house-number / address-frequency features, count clipping | 0.9701 | not submitted |
| v6 | + review bug fixes, India pruning floor 0.55, 150k training entities | 0.9712 | 0.962 (#05) |
| v7 | + native words dropped from blocking, 2 teammate features (3-seed ensemble tested: worse) | 0.9718 | not submitted |
| v8 | + learned transliteration dictionary, name/address cleanup, exact-key channel, 3 features (ideas from public repos) | 0.9768 | 0.968 (#06) |
| v9 | + reverse top-1 channel (each pool record's best S1 among the whole S1 file) as candidates + 3 features | 0.9803 | 0.971 (#07) |
| v10 | + 6 IDF-overlap / acronym / name-number-conflict features, 127-leaf LightGBM | 0.9826 (test-like rows: 0.9807) | not submitted |
| v11 | v10 trained at test-like ownerless-record density (`--universe-frac 0.81`) | **0.9820 on test-like rows** (v9 0.9786 on the same rows) | 0.972174 (#10) |
| v12 | v11 + empty-address candidate channel (measured, not used: +9% candidates, −0.0003) | 0.9817 (test-like) | not submitted |
| **final** | v11 blended with a multilingual MiniLM cross-encoder trained on all 1.32M pairs for two epochs | **0.9844 test-like** | **0.976331 (#13)** |

Sections 3–4 describe v2 as originally reviewed; the v3–v6 changes are summarised in §3.7 and §3.8.

This document is self-contained and written for external review. It covers the task, the data facts
we measured, the current pipeline (v2), the evidence behind each design choice, what failed before and why,
and the specific questions we want reviewed.

---

## 1. Task in one paragraph

Three sources of business records (`entity_id`, `business_name`, `business_address`, `country`), no shared IDs.
Source 1 (S1) is deduplicated. For every S1 entity, output all matching records from Source 2 (S2) and
Source 3 (S3) — zero, one or many. Two output files:
- `matching_results.tsv` — final matches; **the only file scored on the leaderboard**.
- `candidate_pairs.tsv` — the exact candidate set fed into the matching model. **Official update: it counts toward
  the final ranking — "the approach that generates a smaller candidate set per Source 1 entity will be ranked higher
  in the final evaluation beyond the public/private leaderboard."** Every match must be inside the candidate set.

**Metric:** macro F0.5 per S1 entity (precision weighted 2x), averaged over all S1 entities. Singletons (no true
matches) score 1.0 for an empty prediction and 0.0 for any prediction. Public leaderboard = subset of test; final
ranking = private leaderboard (remaining test).

**Constraints:** no external data/APIs of any kind (no geocoding, registries, entity-resolution services); final model
must be MIT/Apache-2.0 and ≤ 8B params (ours is a from-scratch LightGBM). Hardware: one Windows laptop, 32 GB RAM,
CPU only (16 logical cores). A zip with code + README + requirements + methodology doc is required.

---

## 2. Data facts we measured (these drove the design)

| | Train | Test |
|---|---|---|
| S1 | 2,206,821 (US 1,323,633 · India 883,188) | 1,732,544 (India 809,986 · US 663,106 · **France 259,452 = 15%**) |
| S2 | 5,034,616 | 4,887,273 |
| S3 | 5,285,603 | 5,082,316 |

- Ground truth: 7,638,365 matched pairs. **Singletons: 5.58%** of S1. Avg true matches per S1 entity:
  **1.67 in S2 + 1.79 in S3 ≈ 3.46**.
- **Zero cross-country matches** in 7.6M true pairs → country is a hard partition.
- ~73–75% of all S2/S3 records are a true match of *some* S1 entity. So at real scale, "distractors" are mostly
  genuine records of *other* businesses (chain branches, same-name businesses) — far more look-alikes than a
  randomly sampled pool contains.
- **France appears only in test** (0 training examples). Text normalization already handles French legal suffixes
  (`sarl, sas, sasu, sa, eurl, sci`) and features are generic string similarities, but the classifier has never seen
  a French pair.
- S2/S3 records frequently have an **empty address**; names contain typos, character corruption (`consu1tants`,
  `b�otech`), concatenations (`whitemartinez` vs `white and martinez`), abbreviations, and Indic scripts
  (Devanagari/Tamil/Bengali/Gujarati) vs Latin transliterations.

---

## 3. Current pipeline (v2)

```
records ─► normalization (text_repr) ─► candidate generation: IDF-weighted word-token cosine, top-K per S1 per source
        ─► 37 pair features (33 string-similarity + 4 blocking-score) ─► LightGBM ─► per-entity 3-threshold rule
           (v4: pruning to ~12 candidates/entity, 52 features, single threshold — see §3.7)
        ─► global one-to-one conflict resolution ─► outputs
```

### 3.1 Normalization (`text_repr.py`)
- NFKC + casefold, punctuation → space, `&` → `and`, whitespace collapse. Script preserved (no ASCII folding);
  Unicode combining marks preserved (a naive `\w` regex shreds Indic words).
- **Transliteration skeleton**: hand-built Devanagari/Bengali/Tamil/Gujarati → Latin mapping so cross-script pairs
  can share tokens (e.g. `रियल एग्रो प्राइवेट लिमिटेड` → `riyl egro praaivet limited`).
- Legal-suffix detection/stripping (US, India, France lists), address leading digits and digit tokens.

### 3.2 Candidate generation (`blocking_idf.py`) — the key change in v2
- Each record → a bag of word tokens: name tokens from the normalized name **and** its skeleton (prefixed `n:`),
  address tokens (prefixed `a:`), length ≥ 2.
- **Per country, per pool (S2 and S3 separately):** fit TF-IDF (sublinear TF) on the pool; drop tokens with
  document frequency > `max_df = 50,000`; L2-normalize; for every S1 record take the **top-K pool records by cosine**
  (`sparse_dot_topn`, multithreaded). K is per source, so candidates/entity = 2K.
- Outputs per pair: `block_score` (cosine), `block_rank` (0 = best within that entity/pool), `block_top_score`.
- Cost: TF-IDF fit ~20–30 s per country-pool; top-K matmul ≈ 8–9 s per 20k S1 rows per country-pool.

### 3.3 Features (`features.py`, `pipeline_v2.py`) — 37 total
- **Name:** char 3-gram Jaccard (raw and skeleton), token Jaccard, token containment both directions, shared token
  count, Levenshtein ratio (raw and skeleton), token-sort ratio, partial ratio, Jaro-Winkler, LCS ratio, length ratio,
  token-count difference, legal-suffix match / both-present, script match, non-Latin-fraction difference,
  exact normalized-key match.
- **Address:** char n-gram Jaccard, token Jaccard, Levenshtein, token-sort, length ratio, leading-digits match,
  digit-token Jaccard, missing-either / missing-both flags, exact address-key match.
- **Cross-field:** name tokens found in the other record's address (both directions). Plus `country_match`,
  `target_source` (S2 vs S3).
- **Blocking-score features:** `block_score`, `block_rank`, `block_top_score`, `block_score_gap` (top − own).
- Top gain importances: `block_score` ≫ `block_rank` > `addr_digit_token_jaccard` > `name_skeleton_levenshtein_ratio`
  > `addr_token_jaccard` > `addr_char_ngram_jaccard` > `name_token_sort_ratio` > …

### 3.4 Model (`model.py`, `train_v2.py`)
- LightGBM binary: num_leaves 31, max_depth 6, lr 0.05, subsample 0.8, colsample 0.8, min_child_samples 20;
  early stopping (50 rounds) on a separate 6k-entity set → best iteration 1,204.
- **Training data = the real candidate distribution:** 54k train S1 entities blocked against the **full** train pools
  (5.0M / 5.3M) with the exact inference blocking; all candidates kept, no negative downsampling
  (1.08M rows, 175k positives ≈ 1 : 5.2).

### 3.5 Decision rule + constraint (`postprocess.py`)
- Per S1 entity, three thresholds swept jointly against real macro F0.5 on validation:
  `tau0` (if the best candidate's prob < tau0 → predict empty; targets singletons), `tau` (include every candidate
  with prob ≥ tau), `tau_fallback` (if none ≥ tau but best ≥ tau_fallback → predict only the best).
  This was the v2 rule (0.3 / 0.7 / 0.7). From v3 on it is a **single threshold** (tau0 = 0, tau_fallback = tau):
  re-running the 3-threshold sweep on v9 validation scores picked 0.3/0.8/0.8 and scored exactly the same (0.9803),
  so the singleton gate adds nothing. v9–v11 use tau = 0.80 (leaderboard probes on v9: 0.75 → 0.970469,
  0.80 → 0.970791, 0.88 → 0.970398).
- **One-to-one constraint** (verified on train: no S2/S3 record is ever matched to more than one S1 entity): greedy
  global conflict resolution keeps only the highest-probability S1 claimant per candidate, applied once after all
  inference batches.

### 3.6 Inference (`run_inference_v2.py`)
Candidate generation for all 1.73M test S1 at once, then features/prediction in 100k-entity batches; thresholds per
batch; conflict resolution globally; official validator run at the end. Measured: 3.0 h (blocking 55 min, 18 batches
~7.3 min each), peak ~24 GB RAM.

### 3.7 v3 and v4 changes

**v3 (submitted as #04, LB 0.927):** after the reviews, added relative-floor candidate pruning (top 3 + anything
≥ 0.5 × the entity's best → 12.6 candidates/entity), pool key-frequency features and within-entity relative features.
Held-out report-half F0.5 = 0.9399 (v2 at the same candidate policy: 0.9321). Post-processing variants (margin fallback,
expected-F0.5-optimal set size) were measured and were worse than the single threshold.

**v4 (validated 0.9563, not yet submitted).** An error analysis of v3 on the report half (TP 30,193 · FP 551 ·
FN inside candidates 1,793 · FN missed by blocking 2,628) showed the blocking misses were dominated by:
native-script S2/S3 names (27%; 24% of India S2 names are in an Indic script, S1 is 100% Latin, and only 4 of 9
scripts were transliterated), website-style names (`visioncarelynn.com`, 20%), empty/placeholder addresses (22%),
zero-padded house numbers (`0023` vs `23`), look-alike digits (`M0dern`, `5tudio`) and state-name variants
(`Tamil Nadu` / `TN` / `தமிழ்நாடு`). Also: S2/S3 inject fake Latin accents (`Çlub`, `FRÀNCE`) that S1 never has.

Changes (`text_repr.py`, `blocking_idf.py`, `features.py`):
- Normalization: Latin-diacritic fold (combining marks U+0300–036F only; Indic signs untouched), zero-width chars
  removed; names: one look-alike digit in a ≥3-letter token → letter, `www`/`com`/`null` dropped; addresses: leading
  zeros stripped, `NULL`/`N/A` removed, hand-authored tables for street types, US/Indian state names (incl. native
  script) → codes, renamed cities.
- Transliteration skeleton for all 9 Indic scripts, derived from Devanagari via the shared Unicode block layout.
- Phonetic codes (Soundex-like consonant classes over skeleton words): `லக்ஷ்மி`/`Laxmi` → 425,
  `प्राइवेट`/`Private` → 1613. Used as `p:` blocking tokens (S1 and non-ASCII pool names only) and as a feature.
- Blocking also keeps single-digit address numbers.
- Features +3 (52 total): phonetic-code Jaccard, ratio and partial ratio on the space-joined name.
- LightGBM round cap 2000 → 5000 (v3 hit the cap; v4 stopped at 2,781). Threshold re-tuned: 0.75.

Blocking ablation (pruned recall ceiling on the 20k held-out entities, full pools; v3 = 0.924 at 12.3 cand/entity):

| Variant | Recall | India | US | Cand/entity |
|---|---|---|---|---|
| new normalization, v3 tokens | 0.9326 | 0.896 | 0.957 | 12.03 |
| + single-digit numbers | 0.9330 | 0.897 | 0.957 | 12.01 |
| + joined-name `j:` token | 0.8976 | 0.860 | 0.923 | 11.31 |
| + phonetic `p:` tokens | 0.9399 | 0.915 | 0.957 | 12.01 |

The whole-name `j:` token **hurt**: a rare exact-name token outweighs everything else, pulling same-name decoys
above the true duplicate (whose name is noisy). It was dropped; joined-name similarity is kept only as a feature.
Final v4: recall ceiling 0.940 at 11.99 candidates/entity; report-half F0.5 **0.9563** (India 0.947, US 0.962).

### 3.8 v5 and v6 changes

Driven by the v4 error analysis (`REVIEW_v4_ERRORS.md`: TP 30,875 · FP 270 · FN in candidates 1,698 · FN missed by
blocking 2,041) and five external reviews; every blocking change was first measured with `_token_ablation.py`
(pruned recall ceiling on the 20k held-out entities, full pools).

**Blocking.**
- *Glued-name segmentation*: pool names that are one long glued token (`visioncarelynn.com`, `calkinxflh`) are
  segmented by dynamic programming into words from the S1 file of the same split (the true entity's own words are
  always in that vocabulary; no external dictionary) and added as ordinary `n:` tokens. ~4.7–5.1% of pool names.
  Recall 0.9402 → 0.9488 at 11.92 candidates/entity. 821 of the 4,138 v4 blocking misses were glued names made
  entirely of the S1 entity's own words.
- *Field-weighted scoring for India*: a joint TF-IDF norm lets one very rare token (a native-script word) swamp a
  pool record's vector, so an identical address barely counts. For India the score is
  0.7·cos(name) + 1.0·cos(address), and a pool record with no address counts its missing address as partial
  agreement (c = 0.5 × name cosine) so empty-address duplicates are not outranked by same-name decoys. India recall
  0.9146 → 0.9347; the US was better with the joint score (0.9572 vs 0.9488), so the US and France keep it.
  Measured variants: 1/1 0.9231, 1/.7 0.8825, .7/1 0.9326, .5/1 0.9252, .35/1 0.9160 (no empty-address term);
  1/1 c.5 0.9227, .7/1 c.5 0.9433, 1/1 c1 0.8534.
- Combined (v5): **0.9573** recall (India 0.9407, US 0.9683) at 12.46 candidates/entity. Phonetic codes on all pool
  records were tested and rejected (0.9405, 14.0 candidates/entity).
- *Per-country pruning floor (v6)*: India's field-weighted scores are flatter, so its relative floor is 0.55
  (others 0.5): India 13.2 → 11.5 candidates/entity for −0.0006 F0.5; total **11.77** candidates/entity (v4: 11.99).

**Text.** Country-aware address canonicalization: France (test only, no labels) was audited on the unlabeled test
files — S2/S3 give the departement (Nord, Pas-de-Calais, Gironde, Loire-Atlantique) where S1 gives the region, write
`ST` for Saint (our US rule turned it into "street"), and abbreviate allee/chemin/cours/quai. Both admin levels map
to one region code; `st` → saint only for France.

**Features (71).** Counts over the whole S1 file of the split (identical definition at train and test): how many S1
businesses share the candidate's name core (suffix-free, spaces removed), how many *other* S1 businesses own that
exact name, the S1 entity's own name frequency, S1 businesses at the candidate's house-number+street and at its full
address, share of the candidate's name words present anywhere in S1 (invented replacement names have none), pool
distinct addresses per name (chain vs one business) and distinct names per address (shared building); an
order-free, street-anchored house number (match, edit distance, log numeric difference, length difference, number +
street word); legal-form family match (Ltd = Limited); name-missing flag. All pool/S1 counts are clipped (8 / 4)
because the US test pools are 0.62× the train pools, which would otherwise shift raw-count features.

**Fixes from code reviews.** Digit Jaccard is −1 (no evidence) when a side has no digits (was 1.0);
LightGBM `bagging_freq=1` (subsample was a no-op); `addr_key` used the house number twice (`18|18`) — now number +
street word; postcodes keep leading zeros (French 01xxx–09xxx); unit/floor numbers are never taken as the house
number.

**Training.** 150k fit entities (was 54k), 6k early-stopping, same 20k validation; best iteration 3,168;
threshold 0.75 (random re-splits of the validation set pick 0.70–0.75 and show ±0.0015 split noise).

| | v4 | v5 | **v6** |
|---|---|---|---|
| Recall ceiling (report+tune) | 0.940 | 0.957 | 0.954 |
| Candidates / entity | 11.99 | 12.46 | **11.77** |
| Report-half macro F0.5 | 0.9563 | 0.9701 | **0.9712** |
| India / US | 0.947 / 0.962 | 0.962 / 0.976 | 0.962 / 0.977 |
| Singletons | 0.956 | 0.966 | 0.978 |

---

## 4. Results

### 4.1 Validation methodology (important)
Validation = a fixed 20k held-out train S1 entities (69,245 true pairs, 1,140 singletons) blocked against the
**full** train pools. This matters: our earlier validation used 1.5M-record *sampled* pools and predicted
F0.5 ≈ 0.78 for a model that scored **0.519** on the leaderboard. Re-scored at full-pool scale, that same v1
model gets **0.486** — i.e. full-pool validation tracks the leaderboard; sampled-pool validation does not.

### 4.2 Current numbers (full-pool validation, 20k entities)

| | v1 (submitted, LB 0.519) | **v2, K=10** | **v2, K=5** |
|---|---|---|---|
| Recall ceiling (true pairs inside candidate set) | 0.326 | **0.938** | ~0.91 |
| Candidates / entity | 12.7 | 20 | 10 |
| **macro F0.5** | 0.486 | **0.934** | **0.928** |
| Singletons | 0.838 | 0.920 | 0.924 |
| Non-singletons | 0.465 | 0.934 | 0.928 |
| India / US | 0.422 / 0.529 | 0.918 / 0.944 | 0.912 / 0.939 |
| Predicted pairs / entity (truth ≈ 3.46) | 1.12 | 3.01 | 2.95 |

(K=5 numbers = the K=10-trained model evaluated on only the candidates with block_rank < 5.)

Leaderboard progression: 0.433 → 0.518 → 0.519 → 0.927 → 0.962 → 0.968 → 0.970791 → 0.972174 → 0.975 → **0.976331**.
The final result is submission #13; see `SOLUTION.md` for the complete final-model analysis.

---

## 5. What failed before and why (so reviewers don't re-suggest it)

1. **Key/token blocking with absolute frequency caps (v1).** Exact name-key / address-key equality + "share a word"
   with caps (`key count ≤ 30`, `token df ≤ 50`) looked fine on sampled pools but at real 5M-record scale the caps
   masked almost every meaningful word → **recall ceiling 0.326**. The classifier already recovered ~99% of the pairs
   blocking gave it; the whole loss was blocking. Measured alternatives on the true pairs:
   - exact name_key only 21%, address_key 37%, either 49–51%;
   - ~100% of true pairs share ≥ 1 word token, but reaching 93% recall with an absolute df cap needs cap ≈ 1,000 →
     ~850 candidates/entity; "k rarest tokens" (k=2) → 0.88–0.91 recall but ~35k candidates/entity.
   - IDF-weighted top-K dominates both: recall@10 = 0.94 at 10 candidates per source.
2. **Validating on sampled pools** overstated precision (0.78 predicted vs 0.52 real) — random distractors are far less
   confusable than real pools where 75% of records belong to other businesses.
3. **Splink Fellegi-Sunter (EM-trained, term-frequency adjusted) as the final scorer:** 0.405 vs LightGBM 0.486 on the
   same candidates — worse; probabilities poorly calibrated for this task. Dropped.
4. **Character n-gram TF-IDF blocking over the full pool:** high recall at small scale, but 3 columns × 2 pools took
   ~17 min just to fit, with long matmuls — impractical versus word tokens (~10× sparser).
5. **Address word-overlap channel on top of key blocking:** raised coverage but its DuckDB join cost was non-linear
   (56 min for one pool at full scale). Superseded by IDF top-K.

---

## 6. Known gaps / ideas we are considering

1. **Remaining ~6% blocking misses** (after v4). Concatenated/website names (`visioncarelynn.com`) still miss —
   a whole-name token was tried and hurt (§3.7); char n-grams for rare name tokens only remain an option.
   Empty-address records with generic names are the other large group.
2. **K choice / candidate-set-size criterion:** K=5 costs 0.006 F0.5 but halves candidates (10 vs 20 per entity).
   A cheaper-than-K cut (e.g. drop candidates with block_score below a floor, or adaptive K by score gap) might keep
   recall with fewer candidates.
3. **France (15% of test, zero training examples).** No labels to validate on. Candidate ideas: check score
   distributions on test France vs India/US for calibration drift; per-country thresholds (can only be tuned for
   India/US); pseudo-labeling high-confidence French pairs.
4. **India trails US** (0.918 vs 0.944) — likely transliteration/script variants and landmark-style addresses.
5. **Entity-level (listwise) features:** e.g. this candidate's model score relative to the best score for the same S1
   entity, number of candidates above a threshold, whether this candidate is also the top candidate from the other
   pool's perspective (mutual best match). Would need a second-stage model.
6. **Ensembling:** several LightGBM models (seeds / different training samples) averaged.
7. **Training scale:** 54k entities of 2.2M available; more could help, memory permitting.
8. **Conflict resolution:** currently greedy; a max-weight bipartite assignment is the principled alternative.

---

## 7. What we'd like from reviewers

Please focus on concrete, testable improvements given ~47 h, one 32 GB CPU laptop, and the rule that smaller
candidate sets rank higher. In particular:

1. **Blocking:** how would you recover the last ~6% of recall *without* growing candidates/entity? Is there a better
   candidate generator than IDF word-token cosine top-K for this data (typos, transliteration, missing addresses)?
2. **Candidate-set size:** best principled way to trade K vs recall vs F0.5 — adaptive K, score floors, or a cheap
   re-ranker before the main model (the spec says candidate_pairs = what the final model scores)?
3. **France domain shift:** what would you do with zero French labels?
4. **Precision:** with truth ≈ 3.46 matches/entity and us predicting ≈ 3.0, where is F0.5 most likely being lost, and
   which listwise/entity-level features or decision rules help most?
5. **Validation:** any flaw you see in full-pool, entity-level validation on a 20k held-out set (e.g. the
   one-to-one conflict resolution behaves differently when only 20k of 2.2M S1 entities compete)?
6. **Anything in §3 you think is wrong or risky** for the final private-leaderboard evaluation or the code audit.

Please answer in this format:
```
## Top 3 changes (ranked by expected F0.5 gain per hour of work)
## Blocking
## Candidate-set size
## France
## Features / model / decision rule
## Validation risks
## Things you'd NOT do (and why)
```
