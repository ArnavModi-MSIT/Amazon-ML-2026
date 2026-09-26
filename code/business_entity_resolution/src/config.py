"""Central paths, column names and constants for the entity-resolution pipeline.

Single source of truth so a schema/path tweak only needs one edit.
"""
from pathlib import Path

# code/business_entity_resolution/src/config.py -> student_resource/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.txt"
THRESHOLDS_PATH = ARTIFACTS_DIR / "thresholds.json"

VALIDATION_FRACTION = 0.2
# Tuned via measured recall/volume sweep on a real 2000-entity sample (see
# plan.md): freq_cap was the actual driver of the long candidate-count tail
# (p95/max), not top_k/threshold as originally assumed. freq_cap=500 gave
# mean=124.5 candidates/entity (extrapolates to ~212M pairs at full scale,
# ~2x the 10^7-10^8 target); these tighter settings gave mean=35.7
# (~60.7M pairs extrapolated) for a recall cost of only 98.58% -> 97.86%.
BLOCKING_TOP_K = 8
BLOCKING_FREQ_CAP = 30
TFIDF_THRESHOLD = 0.2
NEG_RATIO = 5.0

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCHING_RESULTS_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_PAIRS_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]

SEP = "\t"

# Countries actually observed in training data (do NOT use this to filter/gate
# pipeline behavior — test set includes France, which is absent here by design).
TRAIN_COUNTRIES = ("US", "India")

RANDOM_SEED = 42
