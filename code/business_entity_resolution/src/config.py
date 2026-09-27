"""
config.py

This file just holds all the settings for the project in one place.
Paths, thresholds, blocking limits, etc. That way we don't have random
numbers scattered all over the code.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
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
# Random seed - keeping this fixed makes results repeatable
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------
# We hold out 20% of the training entities to check how well the model
# is doing before we run it on the real test data.
VALIDATION_ENTITY_FRACTION = 0.2

# We try each of these probability cutoffs and see which one gives the
# best F0.5 score on the validation set.
THRESHOLD_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# ---------------------------------------------------------------------------
# Blocking settings (this is the "candidate generation" step)
# ---------------------------------------------------------------------------
MIN_TOKEN_LENGTH = 4          # skip very short/noisy words when blocking
NGRAM_SIZE = 4                 # size of character chunks for n-gram blocking
MAX_CANDIDATES_PER_ENTITY = 60  # don't keep more than this many candidates per entity

# If a word/token is shared by more candidate records than this, we just
# skip it as a blocking key - too common to be useful, and expensive to
# process at this scale.
MAX_POSTING_LIST_SIZE = 1000

# These two blocking rules add extra recall but are expensive to run on
# a huge dataset, so we keep them off by default. Can turn on if needed.
ENABLE_NGRAM_BLOCKING = False
ENABLE_ADDR_TOKEN_BLOCKING = False

# ---------------------------------------------------------------------------
# Processing in chunks (needed because the dataset is millions of rows)
# ---------------------------------------------------------------------------
# We process Source-1 entities in batches of this size instead of all at
# once - doing everything in one go runs out of memory on a dataset this
# big.
SOURCE1_CHUNK_SIZE = 50000

# When building the training data, we keep ALL positive (true match)
# pairs, but only keep a limited number of negative (non-match) pairs
# per positive - there are way more negatives than we actually need.
NEGATIVE_SAMPLE_RATIO = 10
MIN_NEGATIVES_PER_CHUNK = 200

# ---------------------------------------------------------------------------
# Model choice
# ---------------------------------------------------------------------------
MODEL_TYPE = "logistic_regression"  # or "random_forest"

# ---------------------------------------------------------------------------
# Expected columns in the input files
# ---------------------------------------------------------------------------
SOURCE_REQUIRED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_REQUIRED_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

# entity_id prefix tells us which source a row belongs to
SOURCE_PREFIX_MAP = {
    "S1-": "source1",
    "S2-": "source2",
    "S3-": "source3",
}