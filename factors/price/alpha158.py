"""Alpha158 价格类因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


def to_wide(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return data[column].unstack("symbol").astype("float32").sort_index()


def calculate_candle_factor(data: pd.DataFrame, operation: str) -> pd.DataFrame:
    open_price = to_wide(data, "adj_open")
    high = to_wide(data, "adj_high")
    low = to_wide(data, "adj_low")
    close = to_wide(data, "adj_close")
    spread = high - low + 1e-12

    if operation == "KMID":
        return (close - open_price) / open_price
    if operation == "KMID2":
        return (close - open_price) / spread
    if operation == "KUP":
        return (high - np.maximum(open_price, close)) / open_price
    if operation == "KUP2":
        return (high - np.maximum(open_price, close)) / spread
    if operation == "KLOW":
        return (np.minimum(open_price, close) - low) / open_price
    if operation == "KLOW2":
        return (np.minimum(open_price, close) - low) / spread
    if operation == "KSFT":
        return (2 * close - high - low) / open_price
    return (2 * close - high - low) / spread


def calculate_price_ratio_factor(data: pd.DataFrame, operation: str) -> pd.DataFrame:
    close = to_wide(data, "adj_close")
    if operation == "LOW0":
        value = to_wide(data, "adj_low")
    elif operation == "HIGH0":
        value = to_wide(data, "adj_high")
    elif operation == "OPEN0":
        value = to_wide(data, "adj_open")
    elif operation == "HIGH0":
        value = to_wide(data, "adj_high")
    else:
        value = to_wide(data, "vwap")
    return value / close


def calculate_volume_factor(
    data: pd.DataFrame,
    operation: str,
    window: int,
) -> pd.DataFrame:
    volume = to_wide(data, "volume")
    if operation == "VMA":
        return volume.rolling(window, min_periods=1).mean() / (volume + 1e-12)

    volume_change = volume - volume.shift(1)
    increase = volume_change.clip(lower=0).rolling(window, min_periods=1).sum()
    decrease = (-volume_change).clip(lower=0).rolling(window, min_periods=1).sum()
    denominator = volume_change.abs().rolling(window, min_periods=1).sum() + 1e-12
    if operation == "VSUMP":
        return increase / denominator
    if operation == "VSUMN":
        return decrease / denominator
    return (increase - decrease) / denominator


def calculate_regression_factor(
    data: pd.DataFrame,
    operation: str,
    window: int,
) -> pd.DataFrame:
    close = to_wide(data, "adj_close")
    if operation == "BETA":
        return rolling_regression(close, window, "slope") / close
    if operation == "RESI":
        return rolling_regression(close, window, "residual") / close

    result = rolling_regression(close, window, "rsquare")
    rolling_std = close.rolling(window, min_periods=1).std()
    return result.mask(np.isclose(rolling_std, 0, atol=2e-5))


def calculate_price_rolling_factor(
    data: pd.DataFrame,
    operation: str,
    window: int,
) -> pd.DataFrame:
    close = to_wide(data, "adj_close")
    if operation == "ROC":
        return close.shift(window) / close
    if operation == "MA":
        return close.rolling(window, min_periods=1).mean() / close
    if operation == "MAX":
        high = to_wide(data, "adj_high")
        return high.rolling(window, min_periods=1).max() / close
    if operation == "MIN":
        low = to_wide(data, "adj_low")
        return low.rolling(window, min_periods=1).min() / close
    if operation == "QTLU":
        return close.rolling(window, min_periods=1).quantile(0.8) / close
    if operation == "QTLD":
        return close.rolling(window, min_periods=1).quantile(0.2) / close
    return close.rolling(window, min_periods=1).rank(pct=True)


def calculate_position_factor(
    data: pd.DataFrame,
    operation: str,
    window: int,
) -> pd.DataFrame:
    low = to_wide(data, "adj_low")
    if operation == "IMIN":
        return low.rolling(window, min_periods=1).apply(
            lambda values: values.argmin() + 1,
            raw=True,
        ) / window

    high = to_wide(data, "adj_high")
    if operation == "IMAX":
        return high.rolling(window, min_periods=1).apply(
            lambda values: values.argmax() + 1,
            raw=True,
        ) / window
    if operation == "RSV":
        close = to_wide(data, "adj_close")
        rolling_low = low.rolling(window, min_periods=1).min()
        rolling_high = high.rolling(window, min_periods=1).max()
        return (close - rolling_low) / (rolling_high - rolling_low + 1e-12)

    index_max = high.rolling(window, min_periods=1).apply(
        lambda values: values.argmax() + 1,
        raw=True,
    )
    index_min = low.rolling(window, min_periods=1).apply(
        lambda values: values.argmin() + 1,
        raw=True,
    )
    return (index_max - index_min) / window


def calculate_count_factor(
    data: pd.DataFrame,
    operation: str,
    window: int,
) -> pd.DataFrame:
    """计算过去窗口内上涨、下跌交易日占比及二者之差。"""
    close = to_wide(data, "adj_close")
    previous = close.shift(1)
    valid = close.notna() & previous.notna()
    up = close.gt(previous).where(valid).astype(float)
    down = close.lt(previous).where(valid).astype(float)
    up_ratio = up.rolling(window, min_periods=1).mean()
    down_ratio = down.rolling(window, min_periods=1).mean()
    if operation == "CNTP":
        return up_ratio
    if operation == "CNTN":
        return down_ratio
    return up_ratio - down_ratio


def calculate_momentum_factor(
    data: pd.DataFrame,
    operation: str,
    window: int,
) -> pd.DataFrame:
    close = to_wide(data, "adj_close")
    change = close - close.shift(1)
    gain = change.clip(lower=0).rolling(window, min_periods=1).sum()
    loss = (-change).clip(lower=0).rolling(window, min_periods=1).sum()
    denominator = change.abs().rolling(window, min_periods=1).sum() + 1e-12
    if operation == "SUMP":
        return gain / denominator
    if operation == "SUMN":
        return loss / denominator
    return (gain - loss) / denominator


def calculate_count_factor(
    data: pd.DataFrame,
    operation: str,
    window: int,
) -> pd.DataFrame:
    close = to_wide(data, "adj_close")
    change = close - close.shift(1)
    increase_ratio = (change > 0).rolling(window, min_periods=1).mean()
    decrease_ratio = (change < 0).rolling(window, min_periods=1).mean()
    if operation == "CNTP":
        return increase_ratio
    if operation == "CNTN":
        return decrease_ratio
    return increase_ratio - decrease_ratio


def calculate_correlation_factor(
    data: pd.DataFrame,
    operation: str,
    window: int,
) -> pd.DataFrame:
    close = to_wide(data, "adj_close")
    volume = to_wide(data, "volume")
    if operation == "CORR":
        return close.rolling(window, min_periods=1).corr(np.log(volume + 1))
    price_ratio = close / close.shift(1)
    volume_ratio = np.log(volume / volume.shift(1) + 1)
    return price_ratio.rolling(window, min_periods=1).corr(volume_ratio)


def rolling_regression(
    values: pd.DataFrame,
    window: int,
    statistic: str,
) -> pd.DataFrame:
    def calculate(sample: np.ndarray) -> float:
        positions = np.arange(1, len(sample) + 1, dtype=float)
        valid = ~np.isnan(sample)
        x = positions[valid]
        y = sample[valid]
        if len(y) < 2:
            return np.nan

        x_centered = x - x.mean()
        y_centered = y - y.mean()
        x_square_sum = np.sum(x_centered * x_centered)
        if x_square_sum == 0:
            return np.nan

        slope = np.sum(x_centered * y_centered) / x_square_sum
        if statistic == "slope":
            return slope
        if statistic == "residual":
            if np.isnan(sample[-1]):
                return np.nan
            intercept = y.mean() - slope * x.mean()
            return sample[-1] - (slope * positions[-1] + intercept)

        y_square_sum = np.sum(y_centered * y_centered)
        if y_square_sum == 0:
            return np.nan
        covariance = np.sum(x_centered * y_centered)
        return covariance**2 / (x_square_sum * y_square_sum)

    return values.rolling(window, min_periods=1).apply(calculate, raw=True)


def format_result(result: pd.DataFrame) -> pd.DataFrame:
    result.index.name = "date"
    result.columns.name = "symbol"
    return result


@register_factor
class Alpha158PriceFactor(BaseFactor):
    """根据因子名和窗口计算 Alpha158 价格类因子。"""

    name = "alpha158_price"
    description = "Alpha158价格类因子"

    CANDLE_FACTORS = {"KMID", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"}
    PRICE_RATIO_FACTORS = {"LOW0", "OPEN0", "HIGH0", "VWAP0"}
    VOLUME_FACTORS = {"VMA", "VSUMP", "VSUMN", "VSUMD"}
    REGRESSION_FACTORS = {"BETA", "RSQR", "RESI"}
    PRICE_ROLLING_FACTORS = {"ROC", "MA", "MAX", "MIN", "QTLU", "QTLD", "RANK"}
    POSITION_FACTORS = {"RSV", "IMAX", "IMIN", "IMXD"}
    MOMENTUM_FACTORS = {"SUMP", "SUMN", "SUMD"}
    COUNT_FACTORS = {"CNTP", "CNTN", "CNTD"}
    CORRELATION_FACTORS = {"CORR", "CORD"}

    FIXED_FACTORS = CANDLE_FACTORS | PRICE_RATIO_FACTORS
    ROLLING_FACTORS = (
        VOLUME_FACTORS
        | REGRESSION_FACTORS
        | PRICE_ROLLING_FACTORS
        | POSITION_FACTORS
        | COUNT_FACTORS
        | MOMENTUM_FACTORS
        | COUNT_FACTORS
        | CORRELATION_FACTORS
    )

    def __init__(self, operation: str, window: int | None = None) -> None:
        self.operation = operation.upper()
        supported = self.FIXED_FACTORS | self.ROLLING_FACTORS
        if self.operation not in supported:
            raise ValueError(f"不支持的Alpha158价格因子: {self.operation}")
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

        if operation in self.CANDLE_FACTORS:
            result = calculate_candle_factor(data, operation)
        elif operation in self.PRICE_RATIO_FACTORS:
            result = calculate_price_ratio_factor(data, operation)
        elif operation in self.VOLUME_FACTORS:
            result = calculate_volume_factor(data, operation, window)
        elif operation in self.REGRESSION_FACTORS:
            result = calculate_regression_factor(data, operation, window)
        elif operation in self.PRICE_ROLLING_FACTORS:
            result = calculate_price_rolling_factor(data, operation, window)
        elif operation in self.POSITION_FACTORS:
            result = calculate_position_factor(data, operation, window)
        elif operation in self.COUNT_FACTORS:
            result = calculate_count_factor(data, operation, window)
        elif operation in self.MOMENTUM_FACTORS:
            result = calculate_momentum_factor(data, operation, window)
        elif operation in self.COUNT_FACTORS:
            result = calculate_count_factor(data, operation, window)
        elif operation in self.CORRELATION_FACTORS:
            result = calculate_correlation_factor(data, operation, window)
        else:
            raise ValueError(f"不支持的Alpha158价格因子: {operation}")
        return format_result(result)
