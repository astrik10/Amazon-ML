"""
blocking.py

This is the "candidate generation" step. Instead of comparing every
Source-1 record against every Source-2/Source-3 record (way too slow),
we use a few simple rules to quickly find records that MIGHT match, and
only compare those.

Rules used:
  - same country
  - shares a word in the business name
  - shares a word in the address (off by default, see config)
  - shares a 4-character chunk of the name (off by default, see config)
  - same postal code

If ANY rule matches, we keep that record as a candidate. This is called
"union" blocking - we'd rather have some extra wrong candidates than
miss a real match.

Two things make this work on a huge dataset (millions of rows) without
running out of memory or taking forever:

1. We build the "index" of Source-2/Source-3 records ONE TIME
   (build_candidate_index) and reuse it every time we need to look up
   candidates, instead of rebuilding it over and over.
2. We process Source-1 records in smaller batches (chunks) instead of
   matching millions of them against the index all in one go - matching
   everything at once creates a huge intermediate table that doesn't
   fit in memory.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

MAX_POSTING_LIST_SIZE = getattr(config, "MAX_POSTING_LIST_SIZE", 5000)
ENABLE_NGRAM_BLOCKING = getattr(config, "ENABLE_NGRAM_BLOCKING", False)
ENABLE_ADDR_TOKEN_BLOCKING = getattr(config, "ENABLE_ADDR_TOKEN_BLOCKING", True)
SOURCE1_CHUNK_SIZE = getattr(config, "SOURCE1_CHUNK_SIZE", 50000)

RECORD_COLUMNS = [
    "entity_id", "normalized_business_name",
    "normalized_business_address", "normalized_country", "postal_code",
]

RULE_WEIGHTS = {"country": 1, "name_tok": 1, "addr_tok": 1, "name_ngram": 1, "postal": 3}


def _explode_tokens(codes, texts, min_length):
    """One row per word in each text, paired with its code."""
    table = pd.DataFrame({"code": codes, "text": texts.fillna("")})
    table["key"] = table["text"].str.split(" ")
    table = table.explode("key")
    table["key"] = table["key"].fillna("")
    table = table[table["key"].str.len() >= min_length]
    return table[["code", "key"]]


def _explode_ngrams(codes, texts, n):
    """One row per n-character chunk of each (space-removed) text."""
    compact_texts = texts.fillna("").str.replace(" ", "", regex=False)

    out_codes = []
    out_keys = []
    for code, text in zip(codes, compact_texts):
        if len(text) < n:
            if text:
                out_codes.append(code)
                out_keys.append(text)
            continue
        for i in range(len(text) - n + 1):
            out_codes.append(code)
            out_keys.append(text[i : i + n])

    return pd.DataFrame({"code": out_codes, "key": out_keys})


def _explode_exact(codes, values):
    """One row per (code, value) where value is not empty - used for
    country and postal code, where the whole value is the key."""
    table = pd.DataFrame({"code": codes, "key": values})
    return table[table["key"].fillna("") != ""]


def _cap_and_compact(rule_table):
    """Drop keys that are shared by too many records (not useful for
    blocking), and shrink the key column with category dtype so we're
    not storing the same string thousands of times over."""
    if rule_table.empty:
        return rule_table

    key_counts = rule_table.groupby("key")["code"].transform("size")
    rule_table = rule_table[key_counts <= MAX_POSTING_LIST_SIZE].copy()
    rule_table["key"] = rule_table["key"].astype("category")
    rule_table["code"] = rule_table["code"].astype(np.int32)
    return rule_table


class CandidateIndex:
    """
    Holds the pre-built blocking tables for the Source-2/Source-3 pool.
    Build this once with build_candidate_index(), then reuse it for
    every batch of Source-1 records you need to find candidates for.
    """

    def __init__(self, entity_ids, rules, num_records):
        self.entity_ids = entity_ids   # array: code -> real entity_id string
        self.rules = rules             # dict: rule name -> (code, key) table
        self.num_records = num_records


def build_candidate_index(source2_df, source3_df) -> CandidateIndex:
    candidate_pool = pd.concat(
        [source2_df[RECORD_COLUMNS], source3_df[RECORD_COLUMNS]], ignore_index=True
    )
    candidate_pool = candidate_pool[
        ~candidate_pool["entity_id"].astype(str).str.startswith("S1-")
    ].reset_index(drop=True)

    entity_ids = candidate_pool["entity_id"].to_numpy()
    codes = np.arange(len(candidate_pool), dtype=np.int32)

    rules = {}
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

    rule_sizes = []
    for rule_name, rule_table in rules.items():
        rule_sizes.append(f"{rule_name}={len(rule_table)}")
    logger.info("Built candidate index over %d records (%s)", len(candidate_pool), ", ".join(rule_sizes))

    return CandidateIndex(entity_ids=entity_ids, rules=rules, num_records=len(candidate_pool))


def generate_candidates(source1_df: pd.DataFrame, candidate_index: CandidateIndex, chunk_size=None) -> pd.DataFrame:
    """
    Find candidates for every Source-1 record in source1_df.

    We process source1_df in batches (chunks) instead of all at once,
    because matching millions of Source-1 records against the candidate
    index in one go creates a huge intermediate table and runs out of
    memory. Each chunk's result is small (just entity_id + a short list
    of candidate ids), so we can safely collect all the chunk results
    together at the end.
    """
    chunk_size = chunk_size or SOURCE1_CHUNK_SIZE

    if len(source1_df) <= chunk_size:
        return _generate_candidates_for_chunk(source1_df, candidate_index)

    all_chunk_results = []
    total_rows = len(source1_df)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size

    chunk_number = 0
    start = 0
    while start < total_rows:
        chunk_number += 1
        chunk_df = source1_df.iloc[start : start + chunk_size]
        chunk_result = _generate_candidates_for_chunk(chunk_df, candidate_index)
        all_chunk_results.append(chunk_result)

        rows_done = min(start + chunk_size, total_rows)
        logger.info("Blocking: processed chunk %d/%d (%d entities so far)", chunk_number, num_chunks, rows_done)

        start += chunk_size

    return pd.concat(all_chunk_results, ignore_index=True)


def _generate_candidates_for_chunk(source1_df: pd.DataFrame, candidate_index: CandidateIndex) -> pd.DataFrame:
    """
    Same as generate_candidates(), but for one chunk. Returns one row
    per Source-1 entity with a list of candidate entity ids, capped at
    MAX_CANDIDATES_PER_ENTITY, keeping the highest-scoring candidates.
    """
    s1 = source1_df[RECORD_COLUMNS].reset_index(drop=True)
    s1_ids = s1["entity_id"].to_numpy()

    s1_rule_tables = {}
    s1_rule_tables["country"] = _explode_exact(s1_ids, s1["normalized_country"])
    s1_rule_tables["name_tok"] = _explode_tokens(s1_ids, s1["normalized_business_name"], config.MIN_TOKEN_LENGTH)
    s1_rule_tables["postal"] = _explode_exact(s1_ids, s1["postal_code"])

    if ENABLE_ADDR_TOKEN_BLOCKING:
        s1_rule_tables["addr_tok"] = _explode_tokens(
            s1_ids, s1["normalized_business_address"], config.MIN_TOKEN_LENGTH
        )
    if ENABLE_NGRAM_BLOCKING:
        s1_rule_tables["name_ngram"] = _explode_ngrams(s1_ids, s1["normalized_business_name"], config.NGRAM_SIZE)

    for rule_name in s1_rule_tables:
        s1_rule_tables[rule_name] = s1_rule_tables[rule_name].rename(columns={"code": "source1_entity_id"})

    hit_tables = []
    for rule_name, s1_table in s1_rule_tables.items():
        candidate_table = candidate_index.rules.get(rule_name)
        if candidate_table is None or candidate_table.empty or s1_table.empty:
            continue

        merged = s1_table.merge(candidate_table, on="key", how="inner")
        merged = merged.rename(columns={"code": "candidate_code"})
        merged["weight"] = RULE_WEIGHTS[rule_name]
        hit_tables.append(merged[["source1_entity_id", "candidate_code", "weight"]])

    if not hit_tables:
        empty_lists = []
        for _ in range(len(s1_ids)):
            empty_lists.append([])
        return pd.DataFrame({"source1_entity_id": s1_ids, "candidate_entity_ids": empty_lists})

    all_hits = pd.concat(hit_tables, ignore_index=True)

    scored = (
        all_hits.groupby(["source1_entity_id", "candidate_code"], sort=False)["weight"]
        .sum()
        .reset_index(name="score")
    )

    # sort so the best candidates for each entity come first, then keep
    # only the top MAX_CANDIDATES_PER_ENTITY
    scored = scored.sort_values(["source1_entity_id", "score"], ascending=[True, False])
    scored["rank"] = scored.groupby("source1_entity_id").cumcount()
    scored = scored[scored["rank"] < config.MAX_CANDIDATES_PER_ENTITY]

    # turn the integer codes back into real entity_id strings
    scored["candidate_entity_id"] = candidate_index.entity_ids[scored["candidate_code"].to_numpy()]

    # just in case - a Source-1 id should never match itself
    scored = scored[scored["source1_entity_id"] != scored["candidate_entity_id"]]

    grouped_candidates = scored.groupby("source1_entity_id")["candidate_entity_id"].apply(lambda ids: sorted(set(ids)))

    result = pd.DataFrame({"source1_entity_id": s1_ids})
    result["candidate_entity_ids"] = result["source1_entity_id"].map(grouped_candidates)
    result["candidate_entity_ids"] = result["candidate_entity_ids"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    total_candidates = 0
    for id_list in result["candidate_entity_ids"]:
        total_candidates += len(id_list)
    avg_candidates = total_candidates / len(result) if len(result) else 0.0

    logger.info(
        "Generated candidates for %d Source-1 entities (avg %.1f candidates/entity, %d total pairs)",
        len(result), avg_candidates, total_candidates,
    )
    return result


def explode_candidate_pairs(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Turn the one-row-per-entity table into one row per (entity, candidate) pair."""
    exploded = candidates_df.explode("candidate_entity_ids")
    exploded = exploded.rename(columns={"candidate_entity_ids": "candidate_entity_id"})
    exploded = exploded.dropna(subset=["candidate_entity_id"])
    return exploded.reset_index(drop=True)