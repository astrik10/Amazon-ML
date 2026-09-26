"""
Final test inference: load test data, normalize, generate candidates,
build features, score with the final model (trained on ALL training data),
apply the validation-selected threshold, and produce the two required
output files.
"""
from __future__ import annotations

import logging

import pandas as pd

from src import blocking, config, features, labels, model as model_module, preprocessing

logger = logging.getLogger(__name__)


def train_final_model(
    train_source1_df: pd.DataFrame,
    train_source2_df: pd.DataFrame,
    train_source3_df: pd.DataFrame,
    train_ground_truth_df: pd.DataFrame,
) -> tuple[model_module.EntityMatchModel, features.FeatureBuilder]:
    """
    Train the model used for test inference on ALL available training data
    (not just the train-split used for threshold selection), so the final
    model benefits from every labeled example.
    """
    gt_lookup = labels.build_ground_truth_lookup(train_ground_truth_df)

    candidate_index = blocking.build_candidate_index(train_source2_df, train_source3_df)
    candidates = blocking.generate_candidates(train_source1_df, candidate_index)
    pairs = blocking.explode_candidate_pairs(candidates)
    pairs = labels.attach_pair_labels(pairs, gt_lookup)

    entity_pool = features.build_entity_pool(train_source1_df, train_source2_df, train_source3_df)
    feature_builder = features.FeatureBuilder().fit(entity_pool)

    feature_df = feature_builder.build_features_for_pairs(pairs)

    final_model = model_module.EntityMatchModel()
    if feature_df.empty or feature_df["label"].sum() == 0:
        logger.warning("No positive pairs available to train the final model.")
    else:
        final_model.fit(feature_df, feature_df["label"])
    return final_model, feature_builder


def run_test_inference(
    test_source1_df: pd.DataFrame,
    test_source2_df: pd.DataFrame,
    test_source3_df: pd.DataFrame,
    final_model: model_module.EntityMatchModel,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full test-time pipeline and return
    (matching_results_df, candidate_pairs_df) ready to be written to disk.

    A NEW FeatureBuilder is fit on the test entity pool (test source1+2+3)
    so TF-IDF vocabulary reflects the test corpus. This mirrors what a
    from-scratch test run would see and avoids depending on train-only
    vocabulary that may not cover test terms.
    """
    candidate_index = blocking.build_candidate_index(test_source2_df, test_source3_df)
    candidates = blocking.generate_candidates(test_source1_df, candidate_index)
    pairs = blocking.explode_candidate_pairs(candidates)

    entity_pool = features.build_entity_pool(test_source1_df, test_source2_df, test_source3_df)
    test_feature_builder = features.FeatureBuilder().fit(entity_pool)

    feature_df = test_feature_builder.build_features_for_pairs(pairs)
    if not feature_df.empty:
        feature_df = feature_df.copy()
        feature_df["probability"] = final_model.predict_proba(feature_df)
    else:
        feature_df["probability"] = pd.Series(dtype=float)

    # candidate_pairs.tsv: every candidate actually sent to the model
    candidate_pairs_out = _build_candidate_pairs_output(candidates)

    # matching_results.tsv: candidates whose probability clears the threshold
    matched = feature_df[feature_df["probability"] >= threshold] if not feature_df.empty else feature_df
    matching_results_out = _build_matching_results_output(test_source1_df, matched)

    _validate_outputs_are_consistent(matching_results_out, candidate_pairs_out)

    return matching_results_out, candidate_pairs_out


def _build_candidate_pairs_output(candidates_df: pd.DataFrame) -> pd.DataFrame:
    out = candidates_df.copy()
    out["candidate_entity_ids"] = out["candidate_entity_ids"].apply(
        lambda ids: ",".join(sorted(set(ids)))
    )
    return out[["source1_entity_id", "candidate_entity_ids"]]


def _build_matching_results_output(test_source1_df: pd.DataFrame, matched_pairs_df: pd.DataFrame) -> pd.DataFrame:
    matches_by_entity = (
        matched_pairs_df.groupby("source1_entity_id")["candidate_entity_id"].apply(
            lambda ids: ",".join(sorted(set(ids)))
        )
        if not matched_pairs_df.empty
        else pd.Series(dtype=str)
    )
    rows = []
    for eid in test_source1_df["entity_id"]:
        rows.append({
            "source1_entity_id": eid,
            "matched_entity_ids": matches_by_entity.get(eid, ""),
        })
    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])


def _validate_outputs_are_consistent(matching_results_df: pd.DataFrame, candidate_pairs_df: pd.DataFrame) -> None:
    """Sanity check: every matched ID must appear in that entity's candidate list."""
    candidate_lookup = {
        row.source1_entity_id: set(row.candidate_entity_ids.split(",")) if row.candidate_entity_ids else set()
        for row in candidate_pairs_df.itertuples(index=False)
    }
    for row in matching_results_df.itertuples(index=False):
        if not row.matched_entity_ids:
            continue
        matched_ids = set(row.matched_entity_ids.split(","))
        allowed = candidate_lookup.get(row.source1_entity_id, set())
        stray = matched_ids - allowed
        if stray:
            raise AssertionError(
                f"Internal consistency error: entity {row.source1_entity_id} has matched "
                f"IDs {stray} that are not in its own candidate_pairs list."
            )