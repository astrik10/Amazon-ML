"""
Central configuration for the Business Entity Resolution pipeline.

All paths, thresholds and hyperparameters live here so the rest of the
codebase never hard-codes a path or a magic number.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives in <project_root>/src/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

TRAIN_SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1_PATH = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH = TEST_DIR / "test_source3.tsv"

OUTPUT_DIR = PROJECT_ROOT / "output"
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "entity_resolution_model.joblib"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
# Fraction of Source-1 TRAIN entities held out for validation (entity-level
# split, never a record/pair-level split, to avoid leakage).
VALIDATION_ENTITY_FRACTION = 0.2

# Candidate probability thresholds to sweep during validation. The final
# threshold used at test time is chosen ONLY from validation performance.
THRESHOLD_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------
MIN_TOKEN_LENGTH = 3          # ignore very short/noisy tokens when blocking
NGRAM_SIZE = 4                # character n-gram size used for n-gram blocking
MAX_CANDIDATES_PER_ENTITY = 60  # soft cap; highest-scoring candidates kept

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# "logistic_regression" or "random_forest"
MODEL_TYPE = "logistic_regression"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SOURCE_REQUIRED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_REQUIRED_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

# Maps an entity_id prefix to a logical source name. Used to auto-detect the
# source of a file instead of hard-coding which file is which.
SOURCE_PREFIX_MAP = {
    "S1-": "source1",
    "S2-": "source2",
    "S3-": "source3",
}
