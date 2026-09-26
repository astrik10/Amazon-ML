"""
Multi-stage blocking (candidate generation) -- vectorized AND memory-aware.

Two problems showed up at ~12.5M total records:

1. Runtime -- fixed earlier by using pandas merges instead of per-row
   Python dict loops. Character n-gram blocking stays off by default
   (ENABLE_NGRAM_BLOCKING) since exploding every 4-gram of ~10M candidate
   names is the single most expensive rule.

2. Memory -- the candidate pool (Source-2 + Source-3, ~10.3M rows) was
   being exploded into token/n-gram tables FROM SCRATCH on every call:
   once for the train split, once for the validation split, once again
   to train the final model, and once more for test inference. Same
   ~10.3M rows, rebuilt 4 times. That repeated string-heavy explode is
   what ran the process out of memory and segfaulted.

Fix: split "build the candidate-side index" from "match a Source-1 batch
against it". CandidateIndex.build() does the expensive explode/cap work
ONCE over the (fixed) candidate pool, with entity ids stored as int32
codes and key columns as category dtype (both much cheaper than raw
Python strings repeated millions of times). generate_candidates() then
only explodes the (much smaller) Source-1 side per call and merges it
against the pre-built, already-capped candidate index.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

MAX_POSTING_LIST_SIZE = getattr(config, "MAX_POSTING_LIST_SIZE", 5000)
ENABLE_NGRAM_BLOCKING = getattr(config, "ENABLE_NGRAM_BLOCKING", False)
ENABLE_ADDR_TOKEN_BLOCKING = getattr(config, "ENABLE_ADDR_TOKEN_BLOCKING", True)

_POOL_COLS = [
    "entity_id", "normalized_business_name",
    "normalized_business_address", "normalized_country", "postal_code",
]


def _explode_tokens(codes: np.ndarray, texts: pd.Series, min_len: int) -> pd.DataFrame:
    s = pd.DataFrame({"code": codes, "text": texts.fillna("")})
    s["key"] = s["text"].str.split(" ")
    s = s.explode("key")
    s["key"] = s["key"].fillna("")
    s = s[s["key"].str.len() >= min_len]
    return s[["code", "key"]]


def _explode_ngrams(codes: np.ndarray, texts: pd.Series, n: int) -> pd.DataFrame:
    compact = texts.fillna("").str.replace(" ", "", regex=False)
    rows_code, rows_key = [], []
    for code, text in zip(codes, compact):
        if len(text) < n:
            if text:
                rows_code.append(code)
                rows_key.append(text)
            continue
        for i in range(len(text) - n + 1):
            rows_code.append(code)
            rows_key.append(text[i : i + n])
    return pd.DataFrame({"code": rows_code, "key": rows_key})


def _explode_exact(codes: np.ndarray, values: pd.Series) -> pd.DataFrame:
    s = pd.DataFrame({"code": codes, "key": values})
    return s[s["key"].fillna("") != ""]


def _cap_and_compact(rule_df: pd.DataFrame) -> pd.DataFrame:
    """Drop over-common keys, then shrink the key column to category dtype
    so repeated tokens ("street", "ltd", ...) aren't stored as separate
    Python string objects millions of times over."""
    if rule_df.empty:
        return rule_df
    counts = rule_df.groupby("key")["code"].transform("size")
    rule_df = rule_df[counts <= MAX_POSTING_LIST_SIZE].copy()
    rule_df["key"] = rule_df["key"].astype("category")
    rule_df["code"] = rule_df["code"].astype(np.int32)
    return rule_df


@dataclass
class CandidateIndex:
    """Pre-built, capped blocking rule tables over a fixed candidate pool.
    Build once with build_candidate_index(), then pass to as many
    generate_candidates() calls as needed."""
    entity_ids: np.ndarray  # int code -> original entity_id string
    rules: dict[str, pd.DataFrame]  # rule name -> (code, key) DataFrame
    n_records: int


def build_candidate_index(source2_df: pd.DataFrame, source3_df: pd.DataFrame) -> CandidateIndex:
    candidate_pool = pd.concat(
        [source2_df[_POOL_COLS], source3_df[_POOL_COLS]], ignore_index=True
    )
    candidate_pool = candidate_pool[
        ~candidate_pool["entity_id"].astype(str).str.startswith("S1-")
    ].reset_index(drop=True)

    entity_ids = candidate_pool["entity_id"].to_numpy()
    codes = np.arange(len(candidate_pool), dtype=np.int32)

    rules: dict[str, pd.DataFrame] = {}

    rules["country"] = _cap_and_compact(_explode_exact(codes, candidate_pool["normalized_country"]))
    rules["name_tok"] = _cap_and_compact(
        _explode_tokens(codes, candidate_pool["normalized_business_name"], config.MIN_TOKEN_LENGTH)
    )
    if ENABLE_ADDR_TOKEN_BLOCKING:
        rules["addr_tok"] = _cap_and_compact(
            _explode_tokens(codes, candidate_pool["normalized_business_address"], config.MIN_TOKEN_LENGTH)
        )
    if ENABLE_NGRAM_BLOCKING:
        rules["name_ngram"] = _cap_and_compact(
            _explode_ngrams(codes, candidate_pool["normalized_business_name"], config.NGRAM_SIZE)
        )
    rules["postal"] = _cap_and_compact(_explode_exact(codes, candidate_pool["postal_code"]))

    logger.info(
        "Built candidate index over %d records (%s)",
        len(candidate_pool), ", ".join(f"{k}={len(v)}" for k, v in rules.items()),
    )
    return CandidateIndex(entity_ids=entity_ids, rules=rules, n_records=len(candidate_pool))


_RULE_WEIGHTS = {"country": 1, "name_tok": 1, "addr_tok": 1, "name_ngram": 1, "postal": 3}


def generate_candidates(source1_df: pd.DataFrame, candidate_index: CandidateIndex) -> pd.DataFrame:
    """
    Generate blocking candidates for every Source-1 record in source1_df
    against the pre-built candidate_index. Same output contract as before:
    one row per Source-1 entity, candidate_entity_ids (list[str]), capped
    at MAX_CANDIDATES_PER_ENTITY by descending rule-hit score.
    """
    s1 = source1_df[_POOL_COLS].reset_index(drop=True)
    s1_ids = s1["entity_id"].to_numpy()

    s1_rules: dict[str, pd.DataFrame] = {
        "country": _explode_exact(s1_ids, s1["normalized_country"]),
        "name_tok": _explode_tokens(s1_ids, s1["normalized_business_name"], config.MIN_TOKEN_LENGTH),
        "postal": _explode_exact(s1_ids, s1["postal_code"]),
    }
    if ENABLE_ADDR_TOKEN_BLOCKING:
        s1_rules["addr_tok"] = _explode_tokens(s1_ids, s1["normalized_business_address"], config.MIN_TOKEN_LENGTH)
    if ENABLE_NGRAM_BLOCKING:
        s1_rules["name_ngram"] = _explode_ngrams(s1_ids, s1["normalized_business_name"], config.NGRAM_SIZE)
    for k in s1_rules:
        s1_rules[k] = s1_rules[k].rename(columns={"code": "source1_entity_id"})

    hit_frames = []
    for rule_name, s1_side in s1_rules.items():
        cand_side = candidate_index.rules.get(rule_name)
        if cand_side is None or cand_side.empty or s1_side.empty:
            continue
        merged = s1_side.merge(cand_side, on="key", how="inner")
        merged = merged.rename(columns={"code": "candidate_code"})
        merged["weight"] = _RULE_WEIGHTS[rule_name]
        hit_frames.append(merged[["source1_entity_id", "candidate_code", "weight"]])

    if not hit_frames:
        result = pd.DataFrame({
            "source1_entity_id": s1_ids,
            "candidate_entity_ids": [[] for _ in range(len(s1_ids))],
        })
        return result

    all_hits = pd.concat(hit_frames, ignore_index=True)
    scored = (
        all_hits.groupby(["source1_entity_id", "candidate_code"], sort=False)["weight"]
        .sum()
        .reset_index(name="score")
    )

    scored = scored.sort_values(["source1_entity_id", "score"], ascending=[True, False])
    scored["rank"] = scored.groupby("source1_entity_id").cumcount()
    scored = scored[scored["rank"] < config.MAX_CANDIDATES_PER_ENTITY]

    # decode candidate int codes back to original entity_id strings
    scored["candidate_entity_id"] = candidate_index.entity_ids[scored["candidate_code"].to_numpy()]
    # drop accidental self-matches (shouldn't normally happen since pools are
    # disjoint, but keep the guard cheap and explicit)
    scored = scored[scored["source1_entity_id"] != scored["candidate_entity_id"]]

    grouped = scored.groupby("source1_entity_id")["candidate_entity_id"].apply(lambda s: sorted(set(s)))

    result = pd.DataFrame({"source1_entity_id": s1_ids})
    result["candidate_entity_ids"] = result["source1_entity_id"].map(grouped)
    result["candidate_entity_ids"] = result["candidate_entity_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    total_candidates = sum(len(x) for x in result["candidate_entity_ids"])
    avg_candidates = total_candidates / len(result) if len(result) else 0.0
    logger.info(
        "Generated candidates for %d Source-1 entities (avg %.1f candidates/entity, %d total pairs)",
        len(result), avg_candidates, total_candidates,
    )
    return result


def explode_candidate_pairs(candidates_df: pd.DataFrame) -> pd.DataFrame:
    exploded = candidates_df.explode("candidate_entity_ids")
    exploded = exploded.rename(columns={"candidate_entity_ids": "candidate_entity_id"})
    exploded = exploded.dropna(subset=["candidate_entity_id"])
    return exploded.reset_index(drop=True)