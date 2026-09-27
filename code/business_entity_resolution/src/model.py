"""
model.py

Small wrapper around a scikit-learn classifier. We use Logistic
Regression by default (fast, gives nice probabilities), but Random
Forest is available too if you change MODEL_TYPE in config.py.
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
    """Wraps a scikit-learn classifier so the rest of the code doesn't
    need to care which one we're using."""

    def __init__(self, model_type=config.MODEL_TYPE, random_state=config.RANDOM_SEED):
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
        # default: logistic regression
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=self.random_state,
        )

    def fit(self, features_df: pd.DataFrame, labels: pd.Series):
        X = features_df[self.feature_columns].to_numpy(dtype=float)
        y = labels.to_numpy(dtype=int)

        num_positive = int(y.sum())
        num_negative = int(len(y) - num_positive)
        logger.info(
            "Training %s on %d pairs (%d positive, %d negative)",
            self.model_type, len(y), num_positive, num_negative,
        )

        self.model.fit(X, y)
        return self

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        if features_df.empty:
            return np.array([])
        X = features_df[self.feature_columns].to_numpy(dtype=float)
        probabilities = self.model.predict_proba(X)
        return probabilities[:, 1]  # probability of "match" class

    def save(self, path: Path = config.MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(payload, path)
        logger.info("Saved trained model to %s", path)

    @classmethod
    def load(cls, path: Path = config.MODEL_PATH):
        payload = joblib.load(path)
        instance = cls(model_type=payload["model_type"])
        instance.model = payload["model"]
        instance.feature_columns = payload["feature_columns"]
        logger.info("Loaded trained model from %s", path)
        return instance