"""
Classical ML model wrapper.

Uses scikit-learn's LogisticRegression (default) or RandomForestClassifier,
both BSD-licensed and well under the 8B-parameter ceiling. Class imbalance
(most candidate pairs are non-matches) is handled with class_weight="balanced".
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class EntityMatchModel:
    """Thin wrapper around a scikit-learn classifier for pair matching."""

    def __init__(self, model_type: str = config.MODEL_TYPE, random_state: int = config.RANDOM_SEED):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._build_model()
        self.feature_columns = list(FEATURE_COLUMNS)

    def _build_model(self):
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
        # Default: logistic regression baseline.
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=self.random_state,
        )

    def fit(self, features_df: pd.DataFrame, labels: pd.Series) -> "EntityMatchModel":
        X = features_df[self.feature_columns].to_numpy(dtype=float)
        y = labels.to_numpy(dtype=int)
        logger.info(
            "Training %s on %d pairs (%d positive, %d negative)",
            self.model_type, len(y), int(y.sum()), int((1 - y).sum()),
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        if features_df.empty:
            return np.array([])
        X = features_df[self.feature_columns].to_numpy(dtype=float)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path = config.MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "model_type": self.model_type,
                     "feature_columns": self.feature_columns}, path)
        logger.info("Saved trained model to %s", path)

    @classmethod
    def load(cls, path: Path = config.MODEL_PATH) -> "EntityMatchModel":
        payload = joblib.load(path)
        instance = cls(model_type=payload["model_type"])
        instance.model = payload["model"]
        instance.feature_columns = payload["feature_columns"]
        logger.info("Loaded trained model from %s", path)
        return instance
