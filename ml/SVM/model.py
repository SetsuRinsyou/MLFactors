"""Qlib兼容的线性软间隔SVM。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.data.dataset import DataHandlerLP, Dataset
from qlib.model.base import Model
from sklearn.linear_model import SGDClassifier

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from common import make_extreme_labels, preprocess_features


class QlibLinearSVM(Model):
    """以SGD逐epoch优化hinge loss的Qlib预测模型。"""

    def __init__(
        self,
        factor_names: list[str],
        epochs: int = 10,
        alpha: float = 1e-4,
        tail_fraction: float = 0.30,
        mad_scale: float = 5.0,
        random_state: int = 42,
    ) -> None:
        self.factor_names = list(factor_names)
        self.epochs = int(epochs)
        self.alpha = float(alpha)
        self.tail_fraction = float(tail_fraction)
        self.mad_scale = float(mad_scale)
        self.random_state = int(random_state)
        self.estimator = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=self.alpha,
            fit_intercept=True,
            learning_rate="optimal",
            average=True,
            shuffle=True,
            random_state=self.random_state,
        )
        self.fitted = False

    def fit_epoch(self, features: np.ndarray, labels: np.ndarray) -> None:
        kwargs = {} if self.fitted else {"classes": np.array([-1, 1], dtype=np.int8)}
        self.estimator.partial_fit(features, labels, **kwargs)
        self.fitted = True

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("模型尚未训练")
        return self.estimator.decision_function(features).astype(np.float64)

    def fit(self, dataset: Dataset, reweighter=None) -> None:
        frame = dataset.prepare(
            "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        features = preprocess_features(frame["feature"], self.mad_scale)
        labels = make_extreme_labels(
            frame["label"]["fwd_excess_return_5d"], self.tail_fraction
        )
        selected = labels != 0
        for _ in range(self.epochs):
            self.fit_epoch(features[selected], labels[selected])

    def predict(self, dataset: Dataset, segment="test") -> pd.Series:
        features = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        values = preprocess_features(features, self.mad_scale)
        return pd.Series(
            self.decision_function(values), index=features.index, name="score"
        )
