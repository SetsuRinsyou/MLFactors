"""基于拐点识别的SLP斜率反转因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class SLPReversal(BaseFactor):
    """
    基于价格分段与拐点识别的斜率反转因子 (SLP)
    寻找近期波段的极值点（拐点），计算从极值点到当前的斜率，以此判断超跌/超涨的速度。
    """

    name = "slp_reversal_20d"
    description = "拐点斜率反转因子"

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        close = data["adj_close"].unstack("symbol").astype(float).sort_index()

        # 定义计算单列时间序列近期斜率的函数
        def calc_slp(arr: np.ndarray) -> float:
            if np.isnan(arr).all():
                return np.nan
                
            curr_val = arr[-1]
            max_idx = np.nanargmax(arr)
            min_idx = np.nanargmin(arr)
            
            # 判断最近的拐点是最高点还是最低点
            # 距离当前越近的极值点，越能代表当前的波段趋势
            if max_idx > min_idx:
                ext_idx = max_idx
            else:
                ext_idx = min_idx
                
            ext_val = arr[ext_idx]
            delta_t = len(arr) - 1 - ext_idx
            
            if delta_t == 0:
                return 0.0  # 今天就是拐点，尚未形成斜率
                
            # 计算斜率 SLP
            slp = (curr_val - ext_val) / (ext_val * delta_t)
            
            # 因子值取负，因为我们要买入超跌（负斜率极大）的股票
            return -slp

        # 对每只股票进行滚动计算
        # 使用 raw=True 传递 numpy array 以提升运算速度
        signals = close.rolling(self.window, min_periods=5).apply(calc_slp, raw=True)

        return signals
