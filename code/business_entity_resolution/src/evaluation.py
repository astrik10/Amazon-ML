"""
evaluation.py

This is the validation step. We split the training entities into a
"train" part and a "validation" part, train the model on the train
part, then check how well it does on the validation part using the
F0.5 score (same metric the challenge uses to grade us).

We also pick the best probability threshold here - the cutoff we use to
decide "this candidate counts as a match".

Note on scale: with millions of training entities, we can't just build
every single candidate pair and keep them all in memory - there are way
too many (mostly non-matches). So:
  - for TRAINING, we process Source-1 in chunks, and for each chunk we
    keep all the true-match pairs but only keep a limited number of
    non-match pairs (see config.NEGATIVE_SAMPLE_RATIO).
  - for VALIDATION, we still have to score every single candidate pair
    (no shortcuts there, since we need real numbers), but we do it one
    chunk at a time and just add up the running totals instead of
    holding everything in memory at once.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import blocking, config, features, labels, model as model_module

logger = logging.getLogger(__name__)


def f_beta(precision, recall, beta=0.5):
    """F-beta score. With beta=0.5 this rewards precision more than recall."""
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_squared = beta * beta
    denominator = (beta_squared * precision) + recall
    if denominator == 0.0:
        return 0.0
    return (1 + beta_squared) * precision * recall / denominator


def split_entities(entity_ids, fraction, seed):
    """Split entity ids into a train list and a validation list. Using a
    fixed seed means we get the same split every time we run this."""
    rng = np.random.default_rng(seed)
    ids_sorted = np.array(sorted(entity_ids))
    rng.shuffle(ids_sorted)

    num_validation = 0
    if len(ids_sorted) > 1:
        num_validation = max(1, int(round(len(ids_sorted) * fraction)))

    validation_ids = ids_sorted[:num_validation].tolist()
    train_ids = ids_sorted[num_validation:].tolist()
    return train_ids, validation_ids


def compute_entity_level_score(true_matches, predicted_matches):
    """
    F0.5 score for one entity, comparing its true matches against the
    predicted matches. Entities with no true matches (singletons) get a
    special rule: 1.0 if we correctly predicted nothing, 0.0 if we
    wrongly predicted something.
    """
    if not true_matches and not predicted_matches:
        return 1.0
    if not true_matches and predicted_matches:
        return 0.0

    true_positive_count = len(true_matches & predicted_matches)
    precision = true_positive_count / len(predicted_matches) if predicted_matches else 0.0
    recall = true_positive_count / len(true_matches) if true_matches else 0.0
    return f_beta(precision, recall, beta=0.5)


def _iter_chunks(df, chunk_size):
    """Yield df in chunks of chunk_size rows."""
    total_rows = len(df)
    start = 0
    while start < total_rows:
        yield df.iloc[start : start + chunk_size].reset_index(drop=True)
        start += chunk_size


def _update_recall_counters(chunk_s1, chunk_pairs, gt_lookup):
    """For this chunk, count how many true matches exist and how many
    of them actually show up as candidates."""
    if chunk_pairs.empty:
        candidates_by_entity = {}
    else:
        candidates_by_entity = chunk_pairs.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    true_total = 0
    found_total = 0
    for entity_id in chunk_s1["entity_id"]:
        true_matches = gt_lookup.get(entity_id, set())
        if not true_matches:
            continue
        found_matches = candidates_by_entity.get(entity_id, set())
        true_total += len(true_matches)
        found_total += len(true_matches & found_matches)

    return true_total, found_total


def _build_training_set(train_s1, candidate_index, feature_builder, gt_lookup, rng):
    """
    Go through train_s1 in chunks, build candidate pairs, label them,
    keep all positives but only a sample of negatives, and turn what's
    left into features. Returns the combined training features and the
    candidate recall over the whole train split.
    """
    chunk_size = config.SOURCE1_CHUNK_SIZE
    num_chunks = (len(train_s1) + chunk_size - 1) // chunk_size

    feature_chunks = []
    true_total = 0
    found_total = 0
    chunk_number = 0

    for chunk_s1 in _iter_chunks(train_s1, chunk_size):
        chunk_number += 1

        chunk_candidates = blocking.generate_candidates(chunk_s1, candidate_index)
        chunk_pairs = blocking.explode_candidate_pairs(chunk_candidates)

        true_count, found_count = _update_recall_counters(chunk_s1, chunk_pairs, gt_lookup)
        true_total += true_count
        found_total += found_count

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
            "Train set build: chunk %d/%d done (%d entities, %d positives, %d negatives kept)",
            chunk_number, num_chunks, len(chunk_s1), len(positives), len(negatives),
        )

    candidate_recall = (found_total / true_total) if true_total else 1.0

    if feature_chunks:
        train_features = pd.concat(feature_chunks, ignore_index=True)
    else:
        empty_columns = ["source1_entity_id", "candidate_entity_id", "label"] + features.FEATURE_COLUMNS
        train_features = pd.DataFrame(columns=empty_columns)

    return train_features, candidate_recall


def _score_validation_set(val_s1, candidate_index, feature_builder, trained_model, gt_lookup):
    """
    Go through val_s1 in chunks, score every single candidate pair (no
    sampling here - we need the real numbers), and keep a running total
    of pair-level and entity-level stats for each threshold.
    """
    chunk_size = config.SOURCE1_CHUNK_SIZE
    num_chunks = (len(val_s1) + chunk_size - 1) // chunk_size

    true_total = 0
    found_total = 0

    stats_by_threshold = {}
    for threshold in config.THRESHOLD_CANDIDATES:
        stats_by_threshold[threshold] = {
            "tp": 0, "fp": 0, "fn": 0,
            "score_sum": 0.0, "score_count": 0,
            "singleton_total": 0, "singleton_correct": 0,
        }

    chunk_number = 0
    for chunk_s1 in _iter_chunks(val_s1, chunk_size):
        chunk_number += 1

        chunk_candidates = blocking.generate_candidates(chunk_s1, candidate_index)
        chunk_pairs = blocking.explode_candidate_pairs(chunk_candidates)

        true_count, found_count = _update_recall_counters(chunk_s1, chunk_pairs, gt_lookup)
        true_total += true_count
        found_total += found_count

        chunk_pairs = labels.attach_pair_labels(chunk_pairs, gt_lookup)

        if chunk_pairs.empty:
            chunk_scored = chunk_pairs.copy()
            chunk_scored["probability"] = pd.Series(dtype=float)
        else:
            chunk_scored = feature_builder.build_features_for_pairs(chunk_pairs)
            chunk_scored["probability"] = trained_model.predict_proba(chunk_scored)

        for threshold in config.THRESHOLD_CANDIDATES:
            stats = stats_by_threshold[threshold]

            if not chunk_scored.empty:
                y_true = chunk_scored["label"].to_numpy()
                y_pred = (chunk_scored["probability"] >= threshold).astype(int).to_numpy()

                stats["tp"] += int(np.sum((y_true == 1) & (y_pred == 1)))
                stats["fp"] += int(np.sum((y_true == 0) & (y_pred == 1)))
                stats["fn"] += int(np.sum((y_true == 1) & (y_pred == 0)))

                matched_rows = chunk_scored[chunk_scored["probability"] >= threshold]
                predicted_by_entity = matched_rows.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
            else:
                predicted_by_entity = {}

            for entity_id in chunk_s1["entity_id"]:
                true_matches = gt_lookup.get(entity_id, set())
                predicted_matches = predicted_by_entity.get(entity_id, set())
                score = compute_entity_level_score(true_matches, predicted_matches)

                stats["score_sum"] += score
                stats["score_count"] += 1

                if not true_matches:
                    stats["singleton_total"] += 1
                    if not predicted_matches:
                        stats["singleton_correct"] += 1

        logger.info("Validation scoring: chunk %d/%d done (%d entities)", chunk_number, num_chunks, len(chunk_s1))

    candidate_recall = (found_total / true_total) if true_total else 1.0
    return candidate_recall, stats_by_threshold


@dataclass
class ValidationResult:
    chosen_threshold: float
    trained_model: model_module.EntityMatchModel
    feature_builder: features.FeatureBuilder
    report: dict = field(default_factory=dict)


def run_validation(source1_df, source2_df, source3_df, ground_truth_df) -> ValidationResult:
    """
    Full validation pipeline:
      1. split training entities into train / validation
      2. build a (bounded-size) training set and train the model
      3. score the validation set for every threshold
      4. pick the threshold with the best entity-level F0.5
    """
    gt_lookup = labels.build_ground_truth_lookup(ground_truth_df)
    all_entity_ids = source1_df["entity_id"].tolist()
    train_ids, val_ids = split_entities(all_entity_ids, config.VALIDATION_ENTITY_FRACTION, config.RANDOM_SEED)
    logger.info("Entity-level split: %d train entities, %d validation entities", len(train_ids), len(val_ids))

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    train_s1 = source1_df[source1_df["entity_id"].isin(train_id_set)].reset_index(drop=True)
    val_s1 = source1_df[source1_df["entity_id"].isin(val_id_set)].reset_index(drop=True)

    # build the candidate index once, reuse for every chunk below
    candidate_index = blocking.build_candidate_index(source2_df, source3_df)

    # fit TF-IDF once on the whole train+val pool
    entity_pool = features.build_entity_pool(source1_df, source2_df, source3_df)
    feature_builder = features.FeatureBuilder().fit(entity_pool)

    rng = np.random.default_rng(config.RANDOM_SEED)

    logger.info("Building bounded training set from %d train entities (chunked)", len(train_s1))
    train_features, train_recall = _build_training_set(train_s1, candidate_index, feature_builder, gt_lookup, rng)

    trained_model = model_module.EntityMatchModel()
    if train_features.empty or train_features["label"].sum() == 0:
        logger.warning("No positive training pairs found - model will just predict the majority class.")
    else:
        trained_model.fit(train_features, train_features["label"])

    logger.info("Scoring %d validation entities (chunked)", len(val_s1))
    val_recall, stats_by_threshold = _score_validation_set(
        val_s1, candidate_index, feature_builder, trained_model, gt_lookup
    )
    logger.info("Candidate recall -- train: %.3f, validation: %.3f", train_recall, val_recall)

    threshold_rows = []
    best_threshold = config.THRESHOLD_CANDIDATES[0]
    best_macro_score = -1.0

    for threshold in config.THRESHOLD_CANDIDATES:
        stats = stats_by_threshold[threshold]

        precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) else 0.0
        recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) else 0.0
        pair_f05 = f_beta(precision, recall, beta=0.5)

        macro_score = (stats["score_sum"] / stats["score_count"]) if stats["score_count"] else 0.0

        singleton_accuracy = None
        if stats["singleton_total"]:
            singleton_accuracy = stats["singleton_correct"] / stats["singleton_total"]

        threshold_rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f0.5": pair_f05,
            "tp": stats["tp"], "fp": stats["fp"], "fn": stats["fn"],
            "macro_entity_f0.5": macro_score,
            "n_entities": stats["score_count"],
            "n_singletons": stats["singleton_total"],
            "singleton_accuracy": singleton_accuracy,
        })

        if macro_score > best_macro_score:
            best_macro_score = macro_score
            best_threshold = threshold

    report = {
        "train_entities": len(train_ids),
        "val_entities": len(val_ids),
        "train_candidate_recall": train_recall,
        "val_candidate_recall": val_recall,
        "threshold_sweep": threshold_rows,
        "chosen_threshold": best_threshold,
        "chosen_macro_entity_f0.5": best_macro_score,
    }

    _print_report(report)

    return ValidationResult(
        chosen_threshold=best_threshold,
        trained_model=trained_model,
        feature_builder=feature_builder,
        report=report,
    )


def _print_report(report):
    print("\n" + "=" * 72)
    print("VALIDATION REPORT")
    print("=" * 72)
    print(f"Train entities: {report['train_entities']}   Validation entities: {report['val_entities']}")
    print(f"Candidate recall  -- train: {report['train_candidate_recall']:.3f}   "
          f"validation: {report['val_candidate_recall']:.3f}")
    print("-" * 72)
    header = f"{'thr':>5} | {'prec':>6} | {'rec':>6} | {'pairF0.5':>9} | {'macroF0.5':>10} | {'singleton_acc':>14}"
    print(header)
    print("-" * len(header))
    for row in report["threshold_sweep"]:
        if row["singleton_accuracy"] is not None:
            singleton_str = f"{row['singleton_accuracy']:.3f}"
        else:
            singleton_str = "n/a"
        print(
            f"{row['threshold']:.2f}  | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f0.5']:.6f} | {row['macro_entity_f0.5']:.8f} | {singleton_str:>14}"
        )
    print("-" * 72)
    print(f"Chosen threshold: {report['chosen_threshold']:.2f} "
          f"(macro entity F0.5 = {report['chosen_macro_entity_f0.5']:.4f})")
    print("=" * 72 + "\n")