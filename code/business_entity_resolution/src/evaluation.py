"""
Validation: entity-level train/validation split, candidate recall,
pair-level precision/recall/F0.5, entity-level (macro) F0.5 with singleton
handling, and threshold selection.

The entity-level split is critical: we split on source1_entity_id BEFORE
generating any candidates/features, so no record belonging to a given
Source-1 entity appears in both the train and validation portions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import blocking, config, features, labels, model as model_module

logger = logging.getLogger(__name__)


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """F-beta score. Returns 0.0 if precision and recall are both 0."""
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def split_entities(entity_ids: list[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Deterministically shuffle-split a list of Source-1 entity IDs."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(entity_ids))  # sort first for determinism regardless of input order
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * fraction))) if len(ids) > 1 else 0
    val_ids = ids[:n_val].tolist()
    train_ids = ids[n_val:].tolist()
    return train_ids, val_ids


def compute_candidate_recall(
    candidate_pairs_df: pd.DataFrame, gt_lookup: dict[str, set[str]], entity_ids: list[str]
) -> float:
    """
    Fraction of ground-truth matches (for the given entities) that are
    present in the generated candidate pairs -- i.e. the ceiling on recall
    the ML model could possibly achieve given this blocking strategy.
    """
    candidates_by_entity = (
        candidate_pairs_df.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )
    total_true, total_found = 0, 0
    for eid in entity_ids:
        true_matches = gt_lookup.get(eid, set())
        if not true_matches:
            continue
        found = candidates_by_entity.get(eid, set())
        total_true += len(true_matches)
        total_found += len(true_matches & found)
    if total_true == 0:
        return 1.0
    return total_found / total_true


def compute_pair_level_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f0.5": f_beta(precision, recall, beta=0.5),
        "tp": tp, "fp": fp, "fn": fn,
    }


def compute_entity_level_score(true_matches: set[str], predicted_matches: set[str]) -> float:
    """
    Per-Source-1-entity F0.5 with explicit singleton handling:
      - true empty, pred empty     -> 1.0
      - true empty, pred non-empty -> 0.0
      - otherwise standard precision/recall/F0.5 over the two sets
    """
    if not true_matches and not predicted_matches:
        return 1.0
    if not true_matches and predicted_matches:
        return 0.0
    tp = len(true_matches & predicted_matches)
    precision = tp / len(predicted_matches) if predicted_matches else 0.0
    recall = tp / len(true_matches) if true_matches else 0.0
    return f_beta(precision, recall, beta=0.5)


def compute_macro_entity_score(
    scored_pairs: pd.DataFrame, gt_lookup: dict[str, set[str]], entity_ids: list[str], threshold: float
) -> tuple[float, dict]:
    """
    Apply `threshold` to scored_pairs (must contain source1_entity_id,
    candidate_entity_id, probability), build predicted match sets, and
    return the macro-averaged entity-level F0.5 plus a small breakdown.
    """
    predicted = (
        scored_pairs[scored_pairs["probability"] >= threshold]
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )
    per_entity_scores = []
    n_singletons_correct = 0
    n_singletons_total = 0
    for eid in entity_ids:
        true_matches = gt_lookup.get(eid, set())
        pred_matches = predicted.get(eid, set())
        score = compute_entity_level_score(true_matches, pred_matches)
        per_entity_scores.append(score)
        if not true_matches:
            n_singletons_total += 1
            if not pred_matches:
                n_singletons_correct += 1

    macro_score = float(np.mean(per_entity_scores)) if per_entity_scores else 0.0
    breakdown = {
        "n_entities": len(entity_ids),
        "n_singletons": n_singletons_total,
        "singleton_accuracy": (n_singletons_correct / n_singletons_total) if n_singletons_total else None,
    }
    return macro_score, breakdown


@dataclass
class ValidationResult:
    chosen_threshold: float
    trained_model: model_module.EntityMatchModel
    feature_builder: features.FeatureBuilder
    report: dict = field(default_factory=dict)


def run_validation(
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> ValidationResult:
    """
    Full validation pipeline:
      1. entity-level train/val split of Source-1 training entities
      2. candidate generation (blocking) restricted to each split
      3. pair-level labels from ground truth
      4. feature engineering (fit ONCE on train-split entities)
      5. train the model on the train-split pairs
      6. score validation-split pairs, sweep thresholds, pick the one that
         maximizes macro entity-level F0.5
    """
    gt_lookup = labels.build_ground_truth_lookup(ground_truth_df)
    all_entity_ids = source1_df["entity_id"].tolist()
    train_ids, val_ids = split_entities(all_entity_ids, config.VALIDATION_ENTITY_FRACTION, config.RANDOM_SEED)
    logger.info("Entity-level split: %d train entities, %d validation entities", len(train_ids), len(val_ids))

    train_s1 = source1_df[source1_df["entity_id"].isin(train_ids)].reset_index(drop=True)
    val_s1 = source1_df[source1_df["entity_id"].isin(val_ids)].reset_index(drop=True)

    # --- candidate generation ---
    # Source2/Source3 (the candidate pool) is the SAME for both the train
    # split and the val split, so build the (expensive) candidate-side
    # index once and reuse it for both, instead of rebuilding it twice.
    candidate_index = blocking.build_candidate_index(source2_df, source3_df)
    train_candidates = blocking.generate_candidates(train_s1, candidate_index)
    val_candidates = blocking.generate_candidates(val_s1, candidate_index)

    train_pairs = blocking.explode_candidate_pairs(train_candidates)
    val_pairs = blocking.explode_candidate_pairs(val_candidates)

    train_recall = compute_candidate_recall(train_pairs, gt_lookup, train_ids)
    val_recall = compute_candidate_recall(val_pairs, gt_lookup, val_ids)
    logger.info("Candidate recall -- train: %.3f, validation: %.3f", train_recall, val_recall)

    # --- labels ---
    train_pairs = labels.attach_pair_labels(train_pairs, gt_lookup)
    val_pairs = labels.attach_pair_labels(val_pairs, gt_lookup)

    # --- features (fit once on the pool relevant to this validation run) ---
    entity_pool = features.build_entity_pool(source1_df, source2_df, source3_df)
    feature_builder = features.FeatureBuilder().fit(entity_pool)

    train_features = feature_builder.build_features_for_pairs(train_pairs)
    val_features = feature_builder.build_features_for_pairs(val_pairs)

    # --- train ---
    trained_model = model_module.EntityMatchModel()
    if train_features.empty or train_features["label"].sum() == 0:
        logger.warning(
            "No positive training pairs available after blocking -- model will "
            "default to predicting the majority class. Check blocking recall."
        )
    trained_model.fit(train_features, train_features["label"]) if not train_features.empty else None

    # --- score validation pairs ---
    if not val_features.empty:
        val_features = val_features.copy()
        val_features["probability"] = trained_model.predict_proba(val_features)
    else:
        val_features["probability"] = pd.Series(dtype=float)

    # --- pair-level metrics + threshold sweep on entity-level macro F0.5 ---
    threshold_rows = []
    best_threshold, best_macro_score = config.THRESHOLD_CANDIDATES[0], -1.0
    for threshold in config.THRESHOLD_CANDIDATES:
        y_pred = (val_features["probability"] >= threshold).astype(int).to_numpy() if not val_features.empty else np.array([])
        y_true = val_features["label"].to_numpy() if not val_features.empty else np.array([])
        pair_metrics = compute_pair_level_metrics(y_true, y_pred) if len(y_true) else {
            "precision": 0.0, "recall": 0.0, "f0.5": 0.0, "tp": 0, "fp": 0, "fn": 0
        }
        macro_score, breakdown = compute_macro_entity_score(val_features, gt_lookup, val_ids, threshold)
        threshold_rows.append({"threshold": threshold, **pair_metrics, "macro_entity_f0.5": macro_score, **breakdown})
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


def _print_report(report: dict) -> None:
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
        singleton_acc = row["singleton_accuracy"]
        singleton_str = f"{singleton_acc:.3f}" if singleton_acc is not None else "n/a"
        print(
            f"{row['threshold']:.2f}  | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f0.5']:.6f} | {row['macro_entity_f0.5']:.8f} | {singleton_str:>14}"
        )
    print("-" * 72)
    print(f"Chosen threshold: {report['chosen_threshold']:.2f} "
          f"(macro entity F0.5 = {report['chosen_macro_entity_f0.5']:.4f})")
    print("=" * 72 + "\n")