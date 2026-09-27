"""
features.py

For every candidate pair (a Source-1 record + a possible match), we
compute a bunch of similarity numbers - how close are the names, the
addresses, do the postal codes match, etc. These numbers are what we
feed into the ML model.

Note: for TF-IDF similarity we compute it for a whole batch of pairs at
once using sparse matrix math, instead of one pair at a time. Doing it
one at a time (calling sklearn's cosine_similarity in a loop) is really
slow when you have millions of pairs, so we batch it. TF-IDF vectors
from scikit-learn are already normalized to length 1, so "cosine
similarity" between two rows is really just their dot product - that's
what lets us do this as one matrix operation instead of a loop.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

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


def _token_set(text: str) -> set:
    if not text:
        return set()
    return set(text.split(" "))


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / len(union)


def _char_similarity(text_a: str, text_b: str) -> float:
    """How many characters do the two strings share, regardless of order."""
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0
    chars_a = set(text_a)
    chars_b = set(text_b)
    return len(chars_a & chars_b) / len(chars_a | chars_b)


class FeatureBuilder:
    """
    Fits TF-IDF on a set of business names/addresses, then can compute
    features for any pair of entities from that set.

    Fit this once per run (once on the train+validation entities, and
    separately on the test entities), then reuse it for every pair.
    """

    def __init__(self):
        self.name_vectorizer = TfidfVectorizer(min_df=1)
        self.addr_vectorizer = TfidfVectorizer(min_df=1)
        self.name_matrix = None
        self.addr_matrix = None
        self.entity_id_to_row = {}
        self.entity_records = {}
        # row we point to when we see an entity_id we don't recognize -
        # it's just a row of zeros, so its similarity with anything is 0
        self.missing_row = 0

    def fit(self, records_df: pd.DataFrame):
        records_df = records_df.drop_duplicates(subset=["entity_id"]).reset_index(drop=True)

        names = records_df["normalized_business_name"].fillna("").tolist()
        addresses = records_df["normalized_business_address"].fillna("").tolist()

        # TfidfVectorizer needs at least one non-empty document
        if not any(names):
            names = ["__empty__"]
        if not any(addresses):
            addresses = ["__empty__"]

        name_matrix = self.name_vectorizer.fit_transform(names)
        addr_matrix = self.addr_vectorizer.fit_transform(addresses)

        # add one extra all-zero row for unknown entity ids
        zero_name_row = sp.csr_matrix((1, name_matrix.shape[1]))
        zero_addr_row = sp.csr_matrix((1, addr_matrix.shape[1]))
        self.name_matrix = sp.vstack([name_matrix, zero_name_row]).tocsr()
        self.addr_matrix = sp.vstack([addr_matrix, zero_addr_row]).tocsr()
        self.missing_row = self.name_matrix.shape[0] - 1

        self.entity_id_to_row = {}
        for i, entity_id in enumerate(records_df["entity_id"]):
            self.entity_id_to_row[entity_id] = i

        self.entity_records = {}
        for row in records_df.itertuples(index=False):
            self.entity_records[row.entity_id] = {
                "name": row.normalized_business_name,
                "address": row.normalized_business_address,
                "country": row.normalized_country,
                "postal": row.postal_code,
                "source": row.source,
            }

        logger.info("FeatureBuilder fit on %d unique entities", len(records_df))
        return self

    def build_features_for_pairs(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        pairs_df needs source1_entity_id and candidate_entity_id columns.
        Returns the same dataframe with all the feature columns added.
        """
        if pairs_df.empty:
            empty_df = pairs_df.copy()
            for col in FEATURE_COLUMNS:
                empty_df[col] = pd.Series(dtype="float64")
            return empty_df

        s1_ids = pairs_df["source1_entity_id"].to_numpy()
        candidate_ids = pairs_df["candidate_entity_id"].to_numpy()
        num_pairs = len(pairs_df)

        # look up the matrix row number for each id (unknown ids fall
        # back to the zero row we added in fit())
        s1_rows = np.zeros(num_pairs, dtype=np.int64)
        candidate_rows = np.zeros(num_pairs, dtype=np.int64)
        for i in range(num_pairs):
            s1_rows[i] = self.entity_id_to_row.get(s1_ids[i], self.missing_row)
            candidate_rows[i] = self.entity_id_to_row.get(candidate_ids[i], self.missing_row)

        # TF-IDF cosine similarity for the whole batch at once (see the
        # module docstring for why we do it this way instead of a loop)
        name_side_a = self.name_matrix[s1_rows]
        name_side_b = self.name_matrix[candidate_rows]
        name_cosine_scores = np.asarray(name_side_a.multiply(name_side_b).sum(axis=1)).ravel()

        addr_side_a = self.addr_matrix[s1_rows]
        addr_side_b = self.addr_matrix[candidate_rows]
        addr_cosine_scores = np.asarray(addr_side_a.multiply(addr_side_b).sum(axis=1)).ravel()

        # now go through every pair and build the rest of the features
        feature_rows = []
        for i in range(num_pairs):
            record_a = self.entity_records.get(s1_ids[i], {})
            record_b = self.entity_records.get(candidate_ids[i], {})

            name_a = record_a.get("name", "")
            name_b = record_b.get("name", "")
            addr_a = record_a.get("address", "")
            addr_b = record_b.get("address", "")
            country_a = record_a.get("country", "")
            country_b = record_b.get("country", "")
            postal_a = record_a.get("postal", "")
            postal_b = record_b.get("postal", "")
            source_b = record_b.get("source", "")

            name_tokens_a = _token_set(name_a)
            name_tokens_b = _token_set(name_b)
            addr_tokens_a = _token_set(addr_a)
            addr_tokens_b = _token_set(addr_b)

            row = {
                "name_exact_match": float(bool(name_a) and name_a == name_b),
                "name_char_similarity": _char_similarity(name_a, name_b),
                "name_levenshtein_similarity": (
                    Levenshtein.normalized_similarity(name_a, name_b) if (name_a or name_b) else 0.0
                ),
                "name_jaro_winkler_similarity": (
                    JaroWinkler.similarity(name_a, name_b) if (name_a or name_b) else 0.0
                ),
                "name_token_jaccard": _jaccard(name_tokens_a, name_tokens_b),
                "name_token_overlap": float(len(name_tokens_a & name_tokens_b)),
                "name_tfidf_cosine": float(name_cosine_scores[i]),
                "name_length_diff": float(abs(len(name_a) - len(name_b))),
                "address_exact_match": float(bool(addr_a) and addr_a == addr_b),
                "address_char_similarity": _char_similarity(addr_a, addr_b),
                "address_levenshtein_similarity": (
                    Levenshtein.normalized_similarity(addr_a, addr_b) if (addr_a or addr_b) else 0.0
                ),
                "address_token_jaccard": _jaccard(addr_tokens_a, addr_tokens_b),
                "address_token_overlap": float(len(addr_tokens_a & addr_tokens_b)),
                "address_tfidf_cosine": float(addr_cosine_scores[i]),
                "address_length_diff": float(abs(len(addr_a) - len(addr_b))),
                "postal_code_match": float(bool(postal_a) and bool(postal_b) and postal_a == postal_b),
                "country_exact_match": float(bool(country_a) and country_a == country_b),
                "is_source3": float(source_b == "source3"),
            }
            feature_rows.append(row)

        feature_df = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
        feature_df = feature_df.fillna(0.0)

        result_df = pd.concat(
            [pairs_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1
        )
        return result_df


def build_entity_pool(source1_df, source2_df, source3_df) -> pd.DataFrame:
    """Stack source1 + source2 + source3 together so we can fit TF-IDF
    on the full vocabulary."""
    cols = [
        "entity_id", "normalized_business_name", "normalized_business_address",
        "normalized_country", "postal_code", "source",
    ]
    return pd.concat([source1_df[cols], source2_df[cols], source3_df[cols]], ignore_index=True)