"""树模型 — LightGBM / XGBoost 封装。

需要安装可选依赖:  pip install -e ".[tree]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from models.base import BaseModel


class TreeModel(BaseModel):
    """LightGBM 或 XGBoost 模型封装。

    Parameters
    ----------
    engine : "lgbm" 或 "xgb"
    params : 模型超参数
    num_rounds : 训练轮数
    early_stopping_rounds : 早停轮数
    cv_splits : 交叉验证折数
    seed : 随机种子
    """

    def __init__(
        self,
        engine: str = "lgbm",
        params: dict | None = None,
        num_rounds: int = 500,
        early_stopping_rounds: int = 50,
        cv_splits: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(params=params, cv_splits=cv_splits, seed=seed)
        self.engine = engine.lower()
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self._feature_names: list[str] | None = None

    def _default_params(self) -> dict:
        if self.engine == "lgbm":
            return {
                "objective": "regression",
                "metric": "mse",
                "learning_rate": 0.05,
                "num_leaves": 63,
                "max_depth": -1,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "seed": self.seed,
            }
        else:  # xgb
            return {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "verbosity": 0,
                "seed": self.seed,
            }

    def _build_model(self):
        pass  # 树模型在 fit 时构建

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> TreeModel:
        merged = {**self._default_params(), **self.params}

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)

        if self.engine == "lgbm":
            import lightgbm as lgb

            dtrain = lgb.Dataset(X, label=y)
            self.model_ = lgb.train(
                merged,
                dtrain,
                num_boost_round=self.num_rounds,
                valid_sets=[dtrain],
                callbacks=[lgb.log_evaluation(period=0)],
            )
            logger.info("LightGBM 训练完成, best_iteration={}", self.model_.best_iteration)
        else:
            import xgboost as xgb

            dtrain = xgb.DMatrix(X, label=y)
            self.model_ = xgb.train(
                merged,
                dtrain,
                num_boost_round=self.num_rounds,
                evals=[(dtrain, "train")],
                verbose_eval=False,
            )
            logger.info("XGBoost 训练完成")

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("模型未训练，请先调用 fit()")

        if self.engine == "lgbm":
            return self.model_.predict(X)
        else:
            import xgboost as xgb
            dmat = xgb.DMatrix(X)
            return self.model_.predict(dmat)

    def get_feature_importance(self) -> pd.Series | None:
        if self.model_ is None:
            return None

        if self.engine == "lgbm":
            imp = self.model_.feature_importance(importance_type="gain")
            names = self._feature_names or self.model_.feature_name()
        else:
            imp_dict = self.model_.get_score(importance_type="gain")
            names = list(imp_dict.keys())
            imp = np.array(list(imp_dict.values()))

        return pd.Series(imp, index=names, name="importance").sort_values(ascending=False)
