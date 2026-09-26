"""
Ground-truth label handling.

Converts train_ground_truth.tsv (one row per Source-1 entity with a
comma-separated matched_entity_ids string) into:
  1) a lookup dict {source1_entity_id: set(matched_ids)}
  2) pair-level 0/1 labels attached to a candidate-pairs table

Empty ground truth must map to an EMPTY set, never to {""}.
"""
from __future__ import annotations

import pandas as pd

_EMPTY: set[str] = set()


def parse_matched_ids(raw: object) -> set[str]:
    """Parse a comma-separated matched_entity_ids string into a set of IDs."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    text = str(raw).strip()
    if not text:
        return set()
    ids = {part.strip() for part in text.split(",")}
    ids.discard("")  # guard against trailing commas / stray empties
    return ids


def build_ground_truth_lookup(ground_truth_df: pd.DataFrame) -> dict[str, set[str]]:
    """Build {source1_entity_id: set(matched_entity_ids)} from ground truth."""
    lookup: dict[str, set[str]] = {}
    for row in ground_truth_df.itertuples(index=False):
        lookup[row.source1_entity_id] = parse_matched_ids(row.matched_entity_ids)
    return lookup


def attach_pair_labels(pairs_df: pd.DataFrame, gt_lookup: dict[str, set[str]]) -> pd.DataFrame:
    """
    Given a flat pairs table (source1_entity_id, candidate_entity_id), attach
    a binary 'label' column: 1 if the candidate is a true match for that
    Source-1 entity, else 0. Source-1 entities absent from ground truth are
    treated as having no matches (label 0 for all their candidates).

    PERF NOTE: this used to be `out.apply(_label, axis=1)`, which calls a
    Python function once per row through pandas' row-apply machinery --
    on a few hundred thousand candidate pairs that adds up to most of the
    pipeline's runtime for something that's really just two dict lookups.
    A plain zip + list comprehension does the same thing without the
    per-row Series construction overhead.
    """
    out = pairs_df.copy()
    s1_ids = out["source1_entity_id"]
    cand_ids = out["candidate_entity_id"]
    out["label"] = [
        1 if cand_id in gt_lookup.get(s1_id, _EMPTY) else 0
        for s1_id, cand_id in zip(s1_ids, cand_ids)
    ]
    return out