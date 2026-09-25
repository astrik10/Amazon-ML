#!/usr/bin/env python3
"""
Standalone validation script for the two submission output files.

Checks:
  1. Every Source-1 test ID appears exactly once in matching_results.tsv
  2. No duplicate Source-1 rows
  3. Every matched ID exists in test Source 2 or Source 3
  4. No S1 IDs appear as matches
  5. No duplicate matched IDs within a row
  6. Every matched ID exists in that entity's candidate_pairs row
  7. Every candidate ID exists in test Source 2/Source 3
  8. No duplicate candidate IDs within a row
  9. Both output files are tab-separated with the exact required columns
  10. Every Source-1 test ID appears exactly once in candidate_pairs.tsv

Usage:
    python3 utils/validate_submission.py \\
        --matching output/matching_results.tsv \\
        --candidate output/candidate_pairs.tsv \\
        --test-dir dataset/test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MATCHING_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]


def _fail(errors: list[str], msg: str) -> None:
    errors.append(msg)


def _split_ids(raw: str) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw).strip() == "":
        return []
    return [x for x in str(raw).split(",") if x != ""]


def validate(matching_path: Path, candidate_path: Path, test_dir: Path) -> list[str]:
    errors: list[str] = []

    # --- load test source files to know the valid universe of IDs ---
    test_s1_path = test_dir / "test_source1.tsv"
    test_s2_path = test_dir / "test_source2.tsv"
    test_s3_path = test_dir / "test_source3.tsv"
    for p in (test_s1_path, test_s2_path, test_s3_path):
        if not p.exists():
            _fail(errors, f"Missing expected test file: {p}")
    if errors:
        return errors

    test_s1 = pd.read_csv(test_s1_path, sep="\t", dtype=str)
    test_s2 = pd.read_csv(test_s2_path, sep="\t", dtype=str)
    test_s3 = pd.read_csv(test_s3_path, sep="\t", dtype=str)

    valid_s1_ids = set(test_s1["entity_id"])
    valid_candidate_ids = set(test_s2["entity_id"]) | set(test_s3["entity_id"])

    # --- load output files, checking separator/columns first ---
    try:
        matching_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as exc:
        _fail(errors, f"Could not read {matching_path} as TSV: {exc}")
        matching_df = None

    try:
        candidate_df = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as exc:
        _fail(errors, f"Could not read {candidate_path} as TSV: {exc}")
        candidate_df = None

    if matching_df is not None and list(matching_df.columns) != MATCHING_COLUMNS:
        _fail(errors, f"{matching_path} columns {list(matching_df.columns)} != required {MATCHING_COLUMNS}")
    if candidate_df is not None and list(candidate_df.columns) != CANDIDATE_COLUMNS:
        _fail(errors, f"{candidate_path} columns {list(candidate_df.columns)} != required {CANDIDATE_COLUMNS}")

    if matching_df is None or candidate_df is None:
        return errors

    # --- Check 1 & 2: every S1 test ID appears exactly once in matching_results ---
    matching_ids = matching_df["source1_entity_id"].tolist()
    if set(matching_ids) != valid_s1_ids:
        missing = valid_s1_ids - set(matching_ids)
        extra = set(matching_ids) - valid_s1_ids
        if missing:
            _fail(errors, f"matching_results.tsv is missing {len(missing)} Source-1 test ID(s), e.g. {list(missing)[:5]}")
        if extra:
            _fail(errors, f"matching_results.tsv has {len(extra)} unknown Source-1 ID(s), e.g. {list(extra)[:5]}")
    dup_matching = matching_df["source1_entity_id"][matching_df["source1_entity_id"].duplicated()]
    if len(dup_matching):
        _fail(errors, f"matching_results.tsv has duplicate source1_entity_id rows: {dup_matching.unique()[:5].tolist()}")

    # --- Check 10: every S1 test ID appears exactly once in candidate_pairs ---
    candidate_ids_col = candidate_df["source1_entity_id"].tolist()
    if set(candidate_ids_col) != valid_s1_ids:
        missing = valid_s1_ids - set(candidate_ids_col)
        extra = set(candidate_ids_col) - valid_s1_ids
        if missing:
            _fail(errors, f"candidate_pairs.tsv is missing {len(missing)} Source-1 test ID(s), e.g. {list(missing)[:5]}")
        if extra:
            _fail(errors, f"candidate_pairs.tsv has {len(extra)} unknown Source-1 ID(s), e.g. {list(extra)[:5]}")
    dup_candidate = candidate_df["source1_entity_id"][candidate_df["source1_entity_id"].duplicated()]
    if len(dup_candidate):
        _fail(errors, f"candidate_pairs.tsv has duplicate source1_entity_id rows: {dup_candidate.unique()[:5].tolist()}")

    # --- build candidate lookup for cross-checks ---
    candidate_lookup: dict[str, list[str]] = {}
    for row in candidate_df.itertuples(index=False):
        ids = _split_ids(row.candidate_entity_ids)
        candidate_lookup[row.source1_entity_id] = ids
        # Check 8: no duplicate candidate IDs within a row
        if len(ids) != len(set(ids)):
            _fail(errors, f"candidate_pairs.tsv row {row.source1_entity_id} has duplicate candidate IDs")
        # Check 4 (candidate side) / general: no S1 IDs as candidates
        s1_in_candidates = [cid for cid in ids if cid.startswith("S1-")]
        if s1_in_candidates:
            _fail(errors, f"candidate_pairs.tsv row {row.source1_entity_id} lists S1 ID(s) as candidates: {s1_in_candidates}")
        # Check 7: every candidate ID exists in test Source2/Source3
        unknown = [cid for cid in ids if cid not in valid_candidate_ids]
        if unknown:
            _fail(errors, f"candidate_pairs.tsv row {row.source1_entity_id} has candidate ID(s) not in test S2/S3: {unknown[:5]}")

    # --- matching_results cross-checks ---
    for row in matching_df.itertuples(index=False):
        matched_ids = _split_ids(row.matched_entity_ids)
        # Check 5: no duplicate matched IDs
        if len(matched_ids) != len(set(matched_ids)):
            _fail(errors, f"matching_results.tsv row {row.source1_entity_id} has duplicate matched IDs")
        # Check 4: no S1 IDs as matches
        s1_in_matches = [mid for mid in matched_ids if mid.startswith("S1-")]
        if s1_in_matches:
            _fail(errors, f"matching_results.tsv row {row.source1_entity_id} lists S1 ID(s) as matches: {s1_in_matches}")
        # Check 3: every matched ID exists in test Source2/Source3
        unknown = [mid for mid in matched_ids if mid not in valid_candidate_ids]
        if unknown:
            _fail(errors, f"matching_results.tsv row {row.source1_entity_id} has matched ID(s) not in test S2/S3: {unknown[:5]}")
        # Check 6: every matched ID exists in that entity's candidate list
        allowed = set(candidate_lookup.get(row.source1_entity_id, []))
        stray = [mid for mid in matched_ids if mid not in allowed]
        if stray:
            _fail(errors, f"matching_results.tsv row {row.source1_entity_id} has matched ID(s) not present in candidate_pairs.tsv: {stray[:5]}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate entity resolution submission files.")
    parser.add_argument("--matching", required=True, type=Path, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, type=Path, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, type=Path, help="Path to dataset/test directory")
    args = parser.parse_args()

    errors = validate(args.matching, args.candidate, args.test_dir)

    if errors:
        print(f"VALIDATION FAILED with {len(errors)} error(s):\n")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        return 1

    print("VALIDATION PASSED: all checks succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
