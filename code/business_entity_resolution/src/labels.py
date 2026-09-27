"""
labels.py

Handles the ground truth file - turning "S1-001, matches S2-1,S2-2"
into something we can use to label candidate pairs as match/no-match.
"""
from __future__ import annotations

import pandas as pd


def parse_matched_ids(raw) -> set:
    """Turn a comma separated string of ids into a set of ids."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()

    text = str(raw).strip()
    if not text:
        return set()

    parts = text.split(",")
    ids = set()
    for part in parts:
        part = part.strip()
        if part:
            ids.add(part)
    return ids


def build_ground_truth_lookup(ground_truth_df: pd.DataFrame) -> dict:
    """
    Build a dictionary like:
        {"S1-001": {"S2-1", "S2-2"}, "S1-002": {"S3-9"}, ...}
    so we can quickly look up the true matches for any Source-1 entity.
    """
    lookup = {}
    for row in ground_truth_df.itertuples(index=False):
        lookup[row.source1_entity_id] = parse_matched_ids(row.matched_entity_ids)
    return lookup


def attach_pair_labels(pairs_df: pd.DataFrame, gt_lookup: dict) -> pd.DataFrame:
    """
    Given a table of (source1_entity_id, candidate_entity_id) pairs, add
    a 'label' column: 1 if that candidate is a true match, 0 otherwise.
    """
    labels_list = []
    for row in pairs_df.itertuples(index=False):
        true_matches = gt_lookup.get(row.source1_entity_id, set())
        if row.candidate_entity_id in true_matches:
            labels_list.append(1)
        else:
            labels_list.append(0)

    result_df = pairs_df.copy()
    result_df["label"] = labels_list
    return result_df