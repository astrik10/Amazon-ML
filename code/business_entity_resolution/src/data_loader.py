"""
Data loading utilities.

Everything here is deliberately defensive: files are read as TSV with
sep="\\t", required columns are validated, and the logical "source"
(source1 / source2 / source3) of a file is inferred from the entity_id
prefix rather than assumed from the file name. Nothing is hard-coded
about row counts or which countries can appear.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when an input file does not match the expected schema."""


def _validate_columns(df: pd.DataFrame, required_columns: list[str], path: Path) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"File '{path}' is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def infer_source_from_ids(entity_ids: pd.Series) -> str:
    """
    Infer the logical source ('source1' / 'source2' / 'source3') for a
    column of entity_ids by looking at the (majority) ID prefix.

    We use majority-vote rather than the first row so a single malformed
    ID does not break source detection.
    """
    prefixes = entity_ids.astype(str).str.slice(0, 3)
    counts = prefixes.value_counts()
    if counts.empty:
        raise DataValidationError("Cannot infer source: entity_id column is empty.")
    top_prefix = counts.idxmax()
    if top_prefix not in config.SOURCE_PREFIX_MAP:
        raise DataValidationError(
            f"Unrecognized entity_id prefix '{top_prefix}'. "
            f"Expected one of {list(config.SOURCE_PREFIX_MAP.keys())}."
        )
    return config.SOURCE_PREFIX_MAP[top_prefix]


def load_source_file(path: Path, expected_source: Optional[str] = None) -> pd.DataFrame:
    """
    Load a single Source TSV file, validate its schema, and tag every row
    with a 'source' column ('source1' / 'source2' / 'source3').

    Parameters
    ----------
    path: path to the .tsv file
    expected_source: if given, raise if the inferred source does not match
        (helps catch e.g. accidentally swapped file paths).
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected input file not found: {path}")

    logger.info("Loading source file: %s", path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    _validate_columns(df, config.SOURCE_REQUIRED_COLUMNS, path)

    inferred_source = infer_source_from_ids(df["entity_id"])
    if expected_source is not None and inferred_source != expected_source:
        raise DataValidationError(
            f"File '{path}' was expected to contain '{expected_source}' records "
            f"but entity_id prefixes indicate '{inferred_source}'."
        )
    df = df.copy()
    df["source"] = inferred_source

    n_dupes = df["entity_id"].duplicated().sum()
    if n_dupes:
        logger.warning("File '%s' contains %d duplicate entity_id values.", path, n_dupes)

    logger.info("Loaded %d rows from %s (source=%s)", len(df), path.name, inferred_source)
    return df


def load_ground_truth(path: Path) -> pd.DataFrame:
    """
    Load train_ground_truth.tsv and validate its schema.

    matched_entity_ids is kept as a raw string column; parsing into a list
    of IDs is handled separately (see labels.parse_matched_ids) because an
    empty string must map to "no matches", not to [""] .
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    logger.info("Loading ground truth file: %s", path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    _validate_columns(df, config.GROUND_TRUTH_REQUIRED_COLUMNS, path)

    n_dupes = df["source1_entity_id"].duplicated().sum()
    if n_dupes:
        logger.warning(
            "Ground truth file '%s' contains %d duplicate source1_entity_id rows.",
            path, n_dupes,
        )

    logger.info("Loaded ground truth for %d Source-1 entities", len(df))
    return df


def load_train_data() -> dict[str, pd.DataFrame]:
    """Load all four training files."""
    return {
        "source1": load_source_file(config.TRAIN_SOURCE1_PATH, expected_source="source1"),
        "source2": load_source_file(config.TRAIN_SOURCE2_PATH, expected_source="source2"),
        "source3": load_source_file(config.TRAIN_SOURCE3_PATH, expected_source="source3"),
        "ground_truth": load_ground_truth(config.TRAIN_GROUND_TRUTH_PATH),
    }


def load_test_data() -> dict[str, pd.DataFrame]:
    """Load all three test files (no ground truth at test time)."""
    return {
        "source1": load_source_file(config.TEST_SOURCE1_PATH, expected_source="source1"),
        "source2": load_source_file(config.TEST_SOURCE2_PATH, expected_source="source2"),
        "source3": load_source_file(config.TEST_SOURCE3_PATH, expected_source="source3"),
    }
