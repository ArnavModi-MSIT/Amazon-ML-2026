"""Ad-hoc smoke test for features.py: compute features for a real true-match
pair and a real non-match pair, sanity-check the numbers point the right way.
Run: python -m src._smoke_test_features
"""
import numpy as np

from . import blocking, config, features, io_utils

rng = np.random.default_rng(config.RANDOM_SEED)

s1 = io_utils.load_source(config.TRAIN_SOURCE1)
s2 = io_utils.load_source(config.TRAIN_SOURCE2)
gt = io_utils.load_ground_truth()

# Find a true match pair (S1 -> some S2 id) with a non-empty match list.
gt_nonempty = gt[gt["matched_entity_ids"].str.contains("S2-", na=False)]
row = gt_nonempty.iloc[0]
s1_id = row["source1_entity_id"]
s2_id = next(i for i in io_utils.split_id_list(row["matched_entity_ids"]) if i.startswith("S2-"))

s1_row_raw = s1[s1["entity_id"] == s1_id]
s2_row_raw = s2[s2["entity_id"] == s2_id]
s1_rep = blocking.add_representations(s1_row_raw).iloc[0].to_dict()
s2_rep = blocking.add_representations(s2_row_raw).iloc[0].to_dict()

print("=== TRUE MATCH PAIR ===")
print("S1 name:", s1_row_raw.iloc[0]["business_name"].encode("unicode_escape").decode())
print("S2 name:", s2_row_raw.iloc[0]["business_name"].encode("unicode_escape").decode())
feats_match = features.build_pair_features(s1_rep, s2_rep, target_source=0)
for k, v in sorted(feats_match.items()):
    print(f"  {k}: {v:.3f}")

# A random, almost-certainly-non-matching S2 row for contrast.
random_s2_idx = rng.integers(0, len(s2))
s2_random_raw = s2.iloc[[random_s2_idx]]
s2_random_rep = blocking.add_representations(s2_random_raw).iloc[0].to_dict()

print()
print("=== RANDOM (LIKELY NON-MATCH) PAIR ===")
print("S1 name:", s1_row_raw.iloc[0]["business_name"].encode("unicode_escape").decode())
print("S2 name:", s2_random_raw.iloc[0]["business_name"].encode("unicode_escape").decode())
feats_nonmatch = features.build_pair_features(s1_rep, s2_random_rep, target_source=0)
for k, v in sorted(feats_nonmatch.items()):
    print(f"  {k}: {v:.3f}")

print()
print("=== Sanity: key similarity features should be higher for the true match ===")
for key in ["name_char_ngram_jaccard", "name_levenshtein_ratio", "name_token_jaccard",
            "addr_char_ngram_jaccard", "addr_token_jaccard"]:
    m, n = feats_match[key], feats_nonmatch[key]
    flag = "OK" if m >= n else "!! UNEXPECTED (non-match scored higher)"
    print(f"  {key}: match={m:.3f} vs non-match={n:.3f}  {flag}")

print()
print("Feature vector length:", len(features.FEATURE_NAMES))
print("Feature names:", features.FEATURE_NAMES)
