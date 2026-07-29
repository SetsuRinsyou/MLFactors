"""Alpha158 风险类因子。"""

from __future__ import annotations

import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


def to_wide(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return data[column].unstack("symbol").astype("float32").sort_index()


def calculate_range_factor(data: pd.DataFrame) -> pd.DataFrame:
    open_price = to_wide(data, "adj_open")
    high = to_wide(data, "adj_high")
    low = to_wide(data, "adj_low")
    return (high - low) / open_price


def calculate_volatility_factor(
    data: pd.DataFrame,
    operation: str,
    window: int,
) -> pd.DataFrame:
    if operation == "VSTD":
        volume = to_wide(data, "volume")
        return volume.rolling(window, min_periods=1).std() / (volume + 1e-12)

    close = to_wide(data, "adj_close")
    if operation == "STD":
        return close.rolling(window, min_periods=1).std() / close

    volume = to_wide(data, "volume")
    if operation == "VSTD":
        return volume.rolling(window, min_periods=1).std() / (volume + 1e-12)

    weighted_change = (close / close.shift(1) - 1).abs() * volume
    return weighted_change.rolling(window, min_periods=1).std() / (
        weighted_change.rolling(window, min_periods=1).mean() + 1e-12
    )


def format_result(result: pd.DataFrame) -> pd.DataFrame:
    result.index.name = "date"
    result.columns.name = "symbol"
    return result


@register_factor
class Alpha158RiskFactor(BaseFactor):
    """根据因子名和窗口计算 Alpha158 风险类因子。"""

    name = "alpha158_risk"
    description = "Alpha158风险类因子"

    RANGE_FACTORS = {"KLEN"}
    VOLATILITY_FACTORS = {"STD", "VSTD", "WVMA"}

    FIXED_FACTORS = RANGE_FACTORS
    ROLLING_FACTORS = VOLATILITY_FACTORS

    def __init__(self, operation: str, window: int | None = None) -> None:
        self.operation = operation.upper()
        supported = self.FIXED_FACTORS | self.ROLLING_FACTORS
        if self.operation not in supported:
            raise ValueError(f"不支持的Alpha158风险因子: {self.operation}")
        if self.operation in self.ROLLING_FACTORS:
            if not isinstance(window, int) or window <= 0:
                raise ValueError(f"{self.operation}必须提供正整数window")
            self.window = window
        else:
            self.window = 1
        self.factor_name = f"{self.operation}_{self.window}d"

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        operation = self.operation
        window = self.window

        if operation in self.RANGE_FACTORS:
            result = calculate_range_factor(data)
        elif operation in self.VOLATILITY_FACTORS:
            result = calculate_volatility_factor(data, operation, window)
        else:
            raise ValueError(f"不支持的Alpha158风险因子: {operation}")
        return format_result(result)
