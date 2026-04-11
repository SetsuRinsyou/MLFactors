"""PyTorch 神经网络模型封装。

需要安装可选依赖:  pip install -e ".[nn]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from models.base import BaseModel


def _ensure_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("请先安装 PyTorch:  pip install -e ".[nn]")


class _MLP:
    """简单 MLP 工厂（延迟导入 torch）。"""

    @staticmethod
    def build(input_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        torch = _ensure_torch()
        import torch.nn as nn

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)


class NNModel(BaseModel):
    """PyTorch MLP 模型封装。

    Parameters
    ----------
    hidden_dims : 隐藏层维度列表
    lr : 初始学习率
    epochs : 最大训练轮数
    batch_size : 批量大小
    early_stopping_patience : 早停耐心
    dropout : Dropout 比例
    params : 其他参数
    cv_splits : 交叉验证折数
    seed : 随机种子
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 1024,
        early_stopping_patience: int = 10,
        dropout: float = 0.1,
        params: dict | None = None,
        cv_splits: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(params=params, cv_splits=cv_splits, seed=seed)
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.dropout = dropout
        self._feature_names: list[str] | None = None

    def _build_model(self):
        pass  # 在 fit 中根据 input_dim 构建

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> NNModel:
        torch = _ensure_torch()
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        # 处理 NaN
        mask = ~(np.isnan(X_arr).any(axis=1) | np.isnan(y_arr.ravel()))
        X_arr, y_arr = X_arr[mask], y_arr[mask]

        input_dim = X_arr.shape[1]
        self.model_ = _MLP.build(input_dim, self.hidden_dims, self.dropout)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = self.model_.to(device)

        # 划分训练/验证集 (最后 20% 做验证)
        split_idx = int(len(X_arr) * 0.8)
        X_train_t = torch.tensor(X_arr[:split_idx], device=device)
        y_train_t = torch.tensor(y_arr[:split_idx], device=device)
        X_val_t = torch.tensor(X_arr[split_idx:], device=device)
        y_val_t = torch.tensor(y_arr[split_idx:], device=device)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model_.train()
            for xb, yb in train_loader:
                pred = self.model_(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 验证
            self.model_.eval()
            with torch.no_grad():
                val_pred = self.model_(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                logger.info("Early stopping at epoch {}, best val_loss={:.6f}", epoch + 1, best_val_loss)
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
            self.model_ = self.model_.to(device)

        self._device = device
        logger.info("NN 训练完成, epochs={}, val_loss={:.6f}", epoch + 1, best_val_loss)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("模型未训练，请先调用 fit()")

        torch = _ensure_torch()

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_arr, device=self._device)
            pred = self.model_(X_t).cpu().numpy().ravel()
        return pred

    def get_feature_importance(self) -> pd.Series | None:
        """使用第一层权重绝对值作为特征重要性近似。"""
        if self.model_ is None:
            return None
        torch = _ensure_torch()
        first_layer = None
        for module in self.model_.modules():
            if isinstance(module, torch.nn.Linear):
                first_layer = module
                break
        if first_layer is None:
            return None
        imp = first_layer.weight.abs().mean(dim=0).detach().cpu().numpy()
        names = self._feature_names or [f"f{i}" for i in range(len(imp))]
        return pd.Series(imp, index=names, name="importance").sort_values(ascending=False)
