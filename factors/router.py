import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor

@register_factor
class Router(BaseFactor):
    """路由因子。"""

    name = "router"
    description = "Router Factor"

    def __init__(self, name:str) -> None:
        self.name = name

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        if self.name not in data.columns:
            raise KeyError(f"{self.name} not found in data columns")

        result = data[self.name].unstack(level="symbol")
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
