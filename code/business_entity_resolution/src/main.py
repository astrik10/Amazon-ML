"""
Main entry point.

Usage:
    python -m src.main

Runs, in order:
  1. Load + normalize training data
  2. Entity-level validation (candidate recall, threshold sweep, report)
  3. Train the final model on ALL training data
  4. Load + normalize test data, generate candidates, score, apply threshold
  5. Write output/matching_results.tsv and output/candidate_pairs.tsv
"""
from __future__ import annotations

import logging
import random
import sys

import numpy as np

from src import config, data_loader, evaluation, inference, preprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_global_seed(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _normalize(df):
    return preprocessing.add_normalized_columns(df)


def main() -> int:
    set_global_seed()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("STEP 1/5: Loading training data")
    try:
        train_data = data_loader.load_train_data()
    except (FileNotFoundError, data_loader.DataValidationError) as exc:
        logger.error("Failed to load training data: %s", exc)
        return 1

    train_source1 = _normalize(train_data["source1"])
    train_source2 = _normalize(train_data["source2"])
    train_source3 = _normalize(train_data["source3"])
    ground_truth = train_data["ground_truth"]

    logger.info("STEP 2/5: Running entity-level validation")
    val_result = evaluation.run_validation(train_source1, train_source2, train_source3, ground_truth)
    chosen_threshold = val_result.chosen_threshold
    logger.info("Selected probability threshold from validation: %.2f", chosen_threshold)

    logger.info("STEP 3/5: Training final model on ALL training data")
    final_model, _final_feature_builder = inference.train_final_model(
        train_source1, train_source2, train_source3, ground_truth
    )
    final_model.save()

    logger.info("STEP 4/5: Loading and normalizing test data")
    try:
        test_data = data_loader.load_test_data()
    except (FileNotFoundError, data_loader.DataValidationError) as exc:
        logger.error("Failed to load test data: %s", exc)
        return 1

    test_source1 = _normalize(test_data["source1"])
    test_source2 = _normalize(test_data["source2"])
    test_source3 = _normalize(test_data["source3"])

    logger.info("STEP 5/5: Running test inference and writing outputs")
    matching_results_df, candidate_pairs_df = inference.run_test_inference(
        test_source1, test_source2, test_source3, final_model, chosen_threshold
    )

    matching_results_df.to_csv(config.MATCHING_RESULTS_PATH, sep="\t", index=False)
    candidate_pairs_df.to_csv(config.CANDIDATE_PAIRS_PATH, sep="\t", index=False)

    logger.info("Wrote %s (%d rows)", config.MATCHING_RESULTS_PATH, len(matching_results_df))
    logger.info("Wrote %s (%d rows)", config.CANDIDATE_PAIRS_PATH, len(candidate_pairs_df))

    n_with_matches = (matching_results_df["matched_entity_ids"] != "").sum()
    logger.info(
        "%d / %d test Source-1 entities received at least one match",
        n_with_matches, len(matching_results_df),
    )

    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
