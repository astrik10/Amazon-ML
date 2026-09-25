"""
Feature engineering for candidate Source-1 / Source-2-or-3 pairs.

All features are numeric and missing-value-safe. Name/address similarity
uses rapidfuzz (MIT licensed) for Levenshtein and Jaro-Winkler style
similarity, plus scikit-learn's TfidfVectorizer + cosine similarity for a
softer, term-weighted signal. No external services are used.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "name_exact_match",
    "name_char_similarity",
    "name_levenshtein_similarity",
    "name_jaro_winkler_similarity",
    "name_token_jaccard",
    "name_token_overlap",
    "name_tfidf_cosine",
    "name_length_diff",
    "address_exact_match",
    "address_char_similarity",
    "address_levenshtein_similarity",
    "address_token_jaccard",
    "address_token_overlap",
    "address_tfidf_cosine",
    "address_length_diff",
    "postal_code_match",
    "country_exact_match",
    "is_source3",
]


def _token_set(text: str) -> set[str]:
    return set(text.split(" ")) if text else set()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _char_similarity(a: str, b: str) -> float:
    """Simple normalized common-character-count similarity (order-agnostic,
    complements the order-sensitive Levenshtein/Jaro-Winkler features)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    return len(set_a & set_b) / len(set_a | set_b)


class FeatureBuilder:
    """
    Fits TF-IDF vectorizers over a corpus of normalized names/addresses and
    computes the full feature vector for arbitrary candidate pairs.

    A single FeatureBuilder instance should be fit ONCE per pipeline run
    (e.g. once for the training split, once for test inference) over the
    Source-1 + Source-2 + Source-3 records relevant to that run, then reused
    for every pair in that run. This avoids leaking test-only vocabulary
    into training and keeps behavior deterministic.
    """

    def __init__(self) -> None:
        self.name_vectorizer = TfidfVectorizer(min_df=1)
        self.addr_vectorizer = TfidfVectorizer(min_df=1)
        self._name_matrix = None
        self._addr_matrix = None
        self._id_to_row: dict[str, int] = {}
        self._records: dict[str, dict] = {}

    def fit(self, records_df: pd.DataFrame) -> "FeatureBuilder":
        """
        records_df must contain: entity_id, normalized_business_name,
        normalized_business_address, normalized_country, postal_code.
        One row per unique entity across source1+source2+source3.
        """
        records_df = records_df.drop_duplicates(subset=["entity_id"]).reset_index(drop=True)

        names = records_df["normalized_business_name"].fillna("").tolist()
        addrs = records_df["normalized_business_address"].fillna("").tolist()

        # TfidfVectorizer requires at least one non-empty document; guard
        # against a degenerate all-empty corpus.
        safe_names = names if any(names) else ["__empty__"]
        safe_addrs = addrs if any(addrs) else ["__empty__"]
        self._name_matrix = self.name_vectorizer.fit_transform(safe_names)
        self._addr_matrix = self.addr_vectorizer.fit_transform(safe_addrs)

        self._id_to_row = {eid: i for i, eid in enumerate(records_df["entity_id"])}
        self._records = {
            row.entity_id: {
                "name": row.normalized_business_name,
                "address": row.normalized_business_address,
                "country": row.normalized_country,
                "postal": row.postal_code,
                "source": row.source,
            }
            for row in records_df.itertuples(index=False)
        }
        logger.info("FeatureBuilder fit on %d unique entities", len(records_df))
        return self

    def _tfidf_cosine(self, matrix, id_a: str, id_b: str) -> float:
        row_a = self._id_to_row.get(id_a)
        row_b = self._id_to_row.get(id_b)
        if row_a is None or row_b is None:
            return 0.0
        sim = cosine_similarity(matrix[row_a], matrix[row_b])
        return float(sim[0, 0])

    def build_pair_features(self, source1_entity_id: str, candidate_entity_id: str) -> dict:
        rec_a = self._records.get(source1_entity_id, {})
        rec_b = self._records.get(candidate_entity_id, {})

        name_a, name_b = rec_a.get("name", ""), rec_b.get("name", "")
        addr_a, addr_b = rec_a.get("address", ""), rec_b.get("address", "")
        country_a, country_b = rec_a.get("country", ""), rec_b.get("country", "")
        postal_a, postal_b = rec_a.get("postal", ""), rec_b.get("postal", "")
        source_b = rec_b.get("source", "")

        name_tokens_a, name_tokens_b = _token_set(name_a), _token_set(name_b)
        addr_tokens_a, addr_tokens_b = _token_set(addr_a), _token_set(addr_b)

        features = {
            "name_exact_match": float(bool(name_a) and name_a == name_b),
            "name_char_similarity": _char_similarity(name_a, name_b),
            "name_levenshtein_similarity": Levenshtein.normalized_similarity(name_a, name_b)
            if (name_a or name_b) else 0.0,
            "name_jaro_winkler_similarity": JaroWinkler.similarity(name_a, name_b)
            if (name_a or name_b) else 0.0,
            "name_token_jaccard": _jaccard(name_tokens_a, name_tokens_b),
            "name_token_overlap": float(len(name_tokens_a & name_tokens_b)),
            "name_tfidf_cosine": self._tfidf_cosine(self._name_matrix, source1_entity_id, candidate_entity_id),
            "name_length_diff": float(abs(len(name_a) - len(name_b))),
            "address_exact_match": float(bool(addr_a) and addr_a == addr_b),
            "address_char_similarity": _char_similarity(addr_a, addr_b),
            "address_levenshtein_similarity": Levenshtein.normalized_similarity(addr_a, addr_b)
            if (addr_a or addr_b) else 0.0,
            "address_token_jaccard": _jaccard(addr_tokens_a, addr_tokens_b),
            "address_token_overlap": float(len(addr_tokens_a & addr_tokens_b)),
            "address_tfidf_cosine": self._tfidf_cosine(self._addr_matrix, source1_entity_id, candidate_entity_id),
            "address_length_diff": float(abs(len(addr_a) - len(addr_b))),
            "postal_code_match": float(bool(postal_a) and bool(postal_b) and postal_a == postal_b),
            "country_exact_match": float(bool(country_a) and country_a == country_b),
            "is_source3": float(source_b == "source3"),
        }
        return features

    def build_features_for_pairs(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        pairs_df must contain source1_entity_id, candidate_entity_id columns.
        Returns a DataFrame with the original columns plus every feature in
        FEATURE_COLUMNS (all numeric, no NaNs).
        """
        if pairs_df.empty:
            empty = pairs_df.copy()
            for col in FEATURE_COLUMNS:
                empty[col] = pd.Series(dtype="float64")
            return empty

        feature_rows = [
            self.build_pair_features(row.source1_entity_id, row.candidate_entity_id)
            for row in pairs_df.itertuples(index=False)
        ]
        feature_df = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
        feature_df = feature_df.fillna(0.0)
        result = pd.concat(
            [pairs_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1
        )
        return result


def build_entity_pool(source1_df: pd.DataFrame, source2_df: pd.DataFrame, source3_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate normalized source1+2+3 records into one pool for fitting a FeatureBuilder."""
    cols = [
        "entity_id", "normalized_business_name", "normalized_business_address",
        "normalized_country", "postal_code", "source",
    ]
    return pd.concat([source1_df[cols], source2_df[cols], source3_df[cols]], ignore_index=True)
