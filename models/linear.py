"""线性模型 — Ridge / Lasso 封装。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.base import BaseModel


class LinearModel(BaseModel):
    """Ridge / Lasso 模型封装。

    Parameters
    ----------
    engine : "ridge" 或 "lasso"
    params : sklearn 模型参数 (如 alpha)
    cv_splits : 交叉验证折数
    seed : 随机种子
    """

    def __init__(
        self,
        engine: str = "ridge",
        params: dict | None = None,
        cv_splits: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(params=params, cv_splits=cv_splits, seed=seed)
        self.engine = engine.lower()
        self._feature_names: list[str] | None = None

    def _default_params(self) -> dict:
        return {"alpha": 1.0}

    def _build_model(self):
        merged = {**self._default_params(), **self.params}
        if self.engine == "ridge":
            from sklearn.linear_model import Ridge
            self.model_ = Ridge(**merged, random_state=self.seed)
        elif self.engine == "lasso":
            from sklearn.linear_model import Lasso
            self.model_ = Lasso(**merged, random_state=self.seed, max_iter=5000)
        else:
            raise ValueError(f"不支持的线性模型: {self.engine}")

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> LinearModel:
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
        self._build_model()
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        return self.model_.predict(X)

    def get_feature_importance(self) -> pd.Series | None:
        if self.model_ is None:
            return None
        coef = self.model_.coef_
        names = self._feature_names or [f"f{i}" for i in range(len(coef))]
        return pd.Series(np.abs(coef), index=names, name="importance").sort_values(ascending=False)
