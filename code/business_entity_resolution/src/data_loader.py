"""
data_loader.py

Just reads the tsv files and makes sure they look right before we use
them. Nothing fancy here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """We raise this if a file doesn't have the columns we expect."""


def _validate_columns(df, required_columns, path):
    missing_columns = []
    for col in required_columns:
        if col not in df.columns:
            missing_columns.append(col)

    if missing_columns:
        raise DataValidationError(
            f"File '{path}' is missing column(s): {missing_columns}. "
            f"Columns found: {list(df.columns)}"
        )


def infer_source_from_ids(entity_ids: pd.Series) -> str:
    """
    Look at the entity_id prefixes (like S1-, S2-, S3-) and figure out
    which source this file belongs to. We use the most common prefix,
    just in case one row has a typo.
    """
    prefixes = entity_ids.astype(str).str.slice(0, 3)
    prefix_counts = prefixes.value_counts()

    if prefix_counts.empty:
        raise DataValidationError("Cannot figure out the source - entity_id column is empty.")

    most_common_prefix = prefix_counts.idxmax()

    if most_common_prefix not in config.SOURCE_PREFIX_MAP:
        raise DataValidationError(
            f"Unknown entity_id prefix '{most_common_prefix}'. "
            f"Expected one of {list(config.SOURCE_PREFIX_MAP.keys())}."
        )

    return config.SOURCE_PREFIX_MAP[most_common_prefix]


def load_source_file(path: Path, expected_source: Optional[str] = None) -> pd.DataFrame:
    """Load one source tsv file and tag it with which source it is."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info("Loading source file: %s", path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    _validate_columns(df, config.SOURCE_REQUIRED_COLUMNS, path)

    inferred_source = infer_source_from_ids(df["entity_id"])
    if expected_source is not None and inferred_source != expected_source:
        raise DataValidationError(
            f"File '{path}' was supposed to be '{expected_source}' but looks "
            f"like '{inferred_source}' based on the entity_id prefixes."
        )

    df = df.copy()
    df["source"] = inferred_source

    duplicate_count = df["entity_id"].duplicated().sum()
    if duplicate_count:
        logger.warning("File '%s' has %d duplicate entity_id values.", path, duplicate_count)

    logger.info("Loaded %d rows from %s (source=%s)", len(df), path.name, inferred_source)
    return df


def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load train_ground_truth.tsv."""
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    logger.info("Loading ground truth file: %s", path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    _validate_columns(df, config.GROUND_TRUTH_REQUIRED_COLUMNS, path)

    duplicate_count = df["source1_entity_id"].duplicated().sum()
    if duplicate_count:
        logger.warning(
            "Ground truth file '%s' has %d duplicate source1_entity_id rows.", path, duplicate_count
        )

    logger.info("Loaded ground truth for %d Source-1 entities", len(df))
    return df


def load_train_data() -> dict:
    """Load source1, source2, source3 and ground truth for training."""
    return {
        "source1": load_source_file(config.TRAIN_SOURCE1_PATH, expected_source="source1"),
        "source2": load_source_file(config.TRAIN_SOURCE2_PATH, expected_source="source2"),
        "source3": load_source_file(config.TRAIN_SOURCE3_PATH, expected_source="source3"),
        "ground_truth": load_ground_truth(config.TRAIN_GROUND_TRUTH_PATH),
    }


def load_test_data() -> dict:
    """Load source1, source2, source3 for testing (no ground truth here)."""
    return {
        "source1": load_source_file(config.TEST_SOURCE1_PATH, expected_source="source1"),
        "source2": load_source_file(config.TEST_SOURCE2_PATH, expected_source="source2"),
        "source3": load_source_file(config.TEST_SOURCE3_PATH, expected_source="source3"),
    }