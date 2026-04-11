"""因子抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseFactor(ABC):
    """所有因子的基类。

    子类必须定义 ``name`` 属性并实现 ``compute`` 方法。
    因子输出统一为 ``pd.Series``，索引为 ``MultiIndex(date, symbol)``。

    Attributes
    ----------
    name : 因子唯一标识名
    description : 因子描述
    category : 因子类别（如 "momentum", "value", "quality" 等）
    """

    name: str = ""
    description: str = ""
    category: str = "custom"

    @abstractmethod
    def compute(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        """计算因子值。

        Parameters
        ----------
        market_data : 行情数据 DataFrame，MultiIndex(date, symbol)
        fundamental_data : 基本面数据 DataFrame，MultiIndex(date, symbol)，可选

        Returns
        -------
        pd.Series，索引为 MultiIndex(date, symbol)，值为因子值
        """
        ...

    def __repr__(self) -> str:
        return f"<Factor: {self.name} ({self.category})>"
