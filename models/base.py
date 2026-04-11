"""ML 模型抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class BaseModel(ABC):
    """所有模型的基类。

    Parameters
    ----------
    params : 模型超参数字典
    cv_splits : 交叉验证折数
    seed : 随机种子
    """

    def __init__(
        self,
        params: dict | None = None,
        cv_splits: int = 5,
        seed: int = 42,
    ) -> None:
        self.params = params or {}
        self.cv_splits = cv_splits
        self.seed = seed
        self.model_ = None

    @abstractmethod
    def _build_model(self):
        """构建底层模型实例。"""
        ...

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> BaseModel:
        """训练模型。"""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """预测。"""
        ...

    def get_feature_importance(self) -> pd.Series | None:
        """返回特征重要性（若支持）。"""
        return None

    def cross_validate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict[str, list[float]]:
        """使用 TimeSeriesSplit 进行交叉验证。

        Returns
        -------
        dict with keys "train_scores", "val_scores" (IC on each fold)
        """
        from scipy import stats

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        train_scores = []
        val_scores = []

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        for train_idx, val_idx in tscv.split(X_arr):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]

            self.fit(
                pd.DataFrame(X_train) if isinstance(X, pd.DataFrame) else X_train,
                pd.Series(y_train) if isinstance(y, pd.Series) else y_train,
            )
            pred_train = self.predict(X_train)
            pred_val = self.predict(X_val)

            if len(y_train) > 2:
                corr, _ = stats.spearmanr(y_train, pred_train)
                train_scores.append(corr)
            if len(y_val) > 2:
                corr, _ = stats.spearmanr(y_val, pred_val)
                val_scores.append(corr)

        return {"train_scores": train_scores, "val_scores": val_scores}
