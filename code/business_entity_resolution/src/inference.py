"""
inference.py

Two jobs here:
  1. train_final_model - train the model we'll actually use, on ALL the
     training data (not just the train-split part used in evaluation.py).
  2. run_test_inference - run that model on the test data and build the
     two output files.

Same chunking idea as evaluation.py: we go through Source-1 in batches
instead of all at once, because the full dataset is too big to build
every candidate pair in memory in one go.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import blocking, config, features, labels, model as model_module

logger = logging.getLogger(__name__)


def _iter_chunks(df, chunk_size):
    total_rows = len(df)
    start = 0
    while start < total_rows:
        yield df.iloc[start : start + chunk_size].reset_index(drop=True)
        start += chunk_size


def train_final_model(train_source1_df, train_source2_df, train_source3_df, train_ground_truth_df):
    """
    Train the model that will actually be used on the test set. Uses
    ALL training entities (unlike the validation-time model in
    evaluation.py, which only sees the train-split portion).
    """
    gt_lookup = labels.build_ground_truth_lookup(train_ground_truth_df)
    candidate_index = blocking.build_candidate_index(train_source2_df, train_source3_df)

    entity_pool = features.build_entity_pool(train_source1_df, train_source2_df, train_source3_df)
    feature_builder = features.FeatureBuilder().fit(entity_pool)

    rng = np.random.default_rng(config.RANDOM_SEED)
    chunk_size = config.SOURCE1_CHUNK_SIZE
    num_chunks = (len(train_source1_df) + chunk_size - 1) // chunk_size

    feature_chunks = []
    chunk_number = 0

    for chunk_s1 in _iter_chunks(train_source1_df, chunk_size):
        chunk_number += 1

        chunk_candidates = blocking.generate_candidates(chunk_s1, candidate_index)
        chunk_pairs = blocking.explode_candidate_pairs(chunk_candidates)
        chunk_pairs = labels.attach_pair_labels(chunk_pairs, gt_lookup)

        positives = chunk_pairs[chunk_pairs["label"] == 1]
        negatives = chunk_pairs[chunk_pairs["label"] == 0]

        num_negatives_to_keep = max(len(positives) * config.NEGATIVE_SAMPLE_RATIO, config.MIN_NEGATIVES_PER_CHUNK)
        num_negatives_to_keep = min(len(negatives), num_negatives_to_keep)
        if len(negatives) > num_negatives_to_keep:
            negatives = negatives.sample(n=num_negatives_to_keep, random_state=int(rng.integers(0, 2**31 - 1)))

        sampled_pairs = pd.concat([positives, negatives], ignore_index=True)
        if not sampled_pairs.empty:
            feature_chunks.append(feature_builder.build_features_for_pairs(sampled_pairs))

        logger.info(
            "Final model training set: chunk %d/%d done (%d entities, %d positives, %d negatives kept)",
            chunk_number, num_chunks, len(chunk_s1), len(positives), len(negatives),
        )

    if feature_chunks:
        feature_df = pd.concat(feature_chunks, ignore_index=True)
    else:
        feature_df = pd.DataFrame(columns=["label"] + features.FEATURE_COLUMNS)

    final_model = model_module.EntityMatchModel()
    if feature_df.empty or feature_df["label"].sum() == 0:
        logger.warning("No positive pairs available to train the final model.")
    else:
        final_model.fit(feature_df, feature_df["label"])

    return final_model, feature_builder


def run_test_inference(test_source1_df, test_source2_df, test_source3_df, final_model, threshold):
    """
    Run the trained model on the test data and build the two output
    tables: matching_results (final matches) and candidate_pairs (the
    candidates that were considered before thresholding).
    """
    candidate_index = blocking.build_candidate_index(test_source2_df, test_source3_df)

    entity_pool = features.build_entity_pool(test_source1_df, test_source2_df, test_source3_df)
    test_feature_builder = features.FeatureBuilder().fit(entity_pool)

    chunk_size = config.SOURCE1_CHUNK_SIZE
    num_chunks = (len(test_source1_df) + chunk_size - 1) // chunk_size

    matching_chunks = []
    candidate_chunks = []
    chunk_number = 0

    for chunk_s1 in _iter_chunks(test_source1_df, chunk_size):
        chunk_number += 1

        chunk_candidates = blocking.generate_candidates(chunk_s1, candidate_index)
        chunk_pairs = blocking.explode_candidate_pairs(chunk_candidates)

        if chunk_pairs.empty:
            chunk_scored = chunk_pairs.copy()
            chunk_scored["probability"] = pd.Series(dtype=float)
        else:
            chunk_scored = test_feature_builder.build_features_for_pairs(chunk_pairs)
            chunk_scored["probability"] = final_model.predict_proba(chunk_scored)

        if not chunk_scored.empty:
            matched_pairs = chunk_scored[chunk_scored["probability"] >= threshold]
        else:
            matched_pairs = chunk_scored

        candidate_chunks.append(_build_candidate_pairs_output(chunk_candidates))
        matching_chunks.append(_build_matching_results_output(chunk_s1, matched_pairs))

        logger.info("Test inference: chunk %d/%d done (%d entities)", chunk_number, num_chunks, len(chunk_s1))

    if candidate_chunks:
        candidate_pairs_out = pd.concat(candidate_chunks, ignore_index=True)
    else:
        candidate_pairs_out = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_ids"])

    if matching_chunks:
        matching_results_out = pd.concat(matching_chunks, ignore_index=True)
    else:
        matching_results_out = pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])

    _validate_outputs_are_consistent(matching_results_out, candidate_pairs_out)

    return matching_results_out, candidate_pairs_out


def _build_candidate_pairs_output(candidates_df):
    out = candidates_df.copy()
    out["candidate_entity_ids"] = out["candidate_entity_ids"].apply(lambda ids: ",".join(sorted(set(ids))))
    return out[["source1_entity_id", "candidate_entity_ids"]]


def _build_matching_results_output(chunk_source1_df, matched_pairs_df):
    if not matched_pairs_df.empty:
        matches_by_entity = (
            matched_pairs_df.groupby("source1_entity_id")["candidate_entity_id"]
            .apply(lambda ids: ",".join(sorted(set(ids))))
        )
    else:
        matches_by_entity = pd.Series(dtype=str)

    rows = []
    for entity_id in chunk_source1_df["entity_id"]:
        rows.append({
            "source1_entity_id": entity_id,
            "matched_entity_ids": matches_by_entity.get(entity_id, ""),
        })

    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])


def _validate_outputs_are_consistent(matching_results_df, candidate_pairs_df):
    """Make sure every matched id was actually one of that entity's candidates."""
    candidate_lookup = {}
    for row in candidate_pairs_df.itertuples(index=False):
        if row.candidate_entity_ids:
            candidate_lookup[row.source1_entity_id] = set(row.candidate_entity_ids.split(","))
        else:
            candidate_lookup[row.source1_entity_id] = set()

    for row in matching_results_df.itertuples(index=False):
        if not row.matched_entity_ids:
            continue

        matched_ids = set(row.matched_entity_ids.split(","))
        allowed_ids = candidate_lookup.get(row.source1_entity_id, set())
        stray_ids = matched_ids - allowed_ids

        if stray_ids:
            raise AssertionError(
                f"Internal consistency error: entity {row.source1_entity_id} has matched "
                f"IDs {stray_ids} that are not in its own candidate_pairs list."
            )