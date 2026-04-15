"""择时类因子评估汇总报告 — 回测引擎由 vectorbt 提供。

与选股类 ``FactorReport`` 对应，面向单只股票时序信号的评估。

主要指标
--------
- direction_win_rate  : 方向胜率（信号方向与收益方向一致的比例）
- annual_return       : 策略绝对年化收益率（vbt Portfolio.annualized_return）
- max_drawdown        : 策略最大回撤（vbt Portfolio.max_drawdown）
- sharpe              : 策略年化夏普比率（vbt Portfolio.sharpe_ratio）
- calmar              : 绝对卡玛比率（vbt Portfolio.calmar_ratio）
- excess_annual_return: 超额年化收益（超额收益合成净值的 vbt 年化）
- excess_calmar       : 超额卡玛比率（超额收益合成净值的 vbt calmar）

vectorbt 替代范围
-----------------
| 功能                 | 替代前                       | 替代后                          |
|---------------------|------------------------------|---------------------------------|
| 策略仓位 → 净值曲线  | calc_nav + calc_strategy_returns | Portfolio.from_orders          |
| 年化收益             | calc_annual_return            | Portfolio.annualized_return     |
| 最大回撤             | calc_max_drawdown             | Portfolio.max_drawdown          |
| 夏普比率             | calc_sharpe                   | Portfolio.sharpe_ratio          |
| 卡玛比率             | calmar = annual / mdd         | Portfolio.calmar_ratio          |
| 超额净值指标         | calc_nav + 同上               | 超额合成净值 Portfolio（同上）  |
| 方向胜率             | calc_direction_win_rate       | 保留（vbt 无逐 bar 方向匹配）   |
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from data.schema import Col
from evaluation.timimg.metrics import (
    calc_direction_win_rate,
    calc_excess_returns,
)


def _year_freq(trading_days: int) -> str:
    """将交易日数转换为 vectorbt year_freq 字符串（例如 ``'252D'``）。"""
    return f"{trading_days}D"


class TimingReport:
    """择时因子评估报告（回测引擎由 vectorbt 提供）。

    Parameters
    ----------
    factor_values : 择时信号 pd.Series，索引为 date
        正值 = 多头方向，负值 = 空头方向，0 = 空仓
    market_data : 行情数据 pd.DataFrame
        MultiIndex(date, symbol) 格式（来自 DataLoader）或 date 单索引 DataFrame
    symbol : 目标股票代码
    benchmark_data : 基准行情数据（可选），格式与 market_data 相同
        ``None`` 时超额收益以 0 为基准（即超额 = 绝对收益）
    benchmark_symbol : 基准股票/指数代码（benchmark_data 为 MultiIndex 时使用）
    price_col : 计算收益使用的价格列名（默认 ``"close"``）
    trading_days : 年化交易日数（默认 252）
    risk_free : 年化无风险利率，用于夏普计算（默认 0.0）

    Examples
    --------
    >>> report = TimingReport(signal, market_data, symbol="600036")
    >>> print(report.summary())
    """

    def __init__(
        self,
        factor_values: pd.Series,
        market_data: pd.DataFrame,
        symbol: str,
        benchmark_data: pd.DataFrame | None = None,
        benchmark_symbol: str | None = None,
        price_col: str = Col.CLOSE,
        trading_days: int = 252,
        risk_free: float = 0.0,
    ) -> None:
        self.factor_values = factor_values
        self.market_data = market_data
        self.symbol = symbol
        self.benchmark_data = benchmark_data
        self.benchmark_symbol = benchmark_symbol
        self.price_col = price_col
        self.trading_days = trading_days
        self.risk_free = risk_free

        # 内部缓存
        self._close_cache: pd.Series | None = None
        self._benchmark_close_cache: pd.Series | None = None
        self._pf = None           # vbt.Portfolio（策略）
        self._bm_pf = None        # vbt.Portfolio（基准买入持有）
        self._excess_pf = None    # vbt.Portfolio（超额合成净值）
        self._summary: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    #  价格提取（私有）
    # ------------------------------------------------------------------ #

    def _get_close(self) -> pd.Series:
        """提取目标股票收盘价序列。"""
        if self._close_cache is not None:
            return self._close_cache

        md = self.market_data
        if isinstance(md.index, pd.MultiIndex):
            level_vals = md.index.get_level_values(Col.SYMBOL)
            key = self.symbol
            if not (level_vals == self.symbol).any():
                try:
                    key = level_vals.dtype.type(self.symbol)
                except (ValueError, TypeError):
                    pass
            try:
                close = md.xs(key, level=Col.SYMBOL)[self.price_col]
            except KeyError:
                raise KeyError(
                    f"market_data 中未找到股票 '{self.symbol}'，"
                    f"请检查 load_data() 时是否包含该股票代码。"
                )
        else:
            close = md[self.price_col]

        close.index = pd.to_datetime(close.index)
        self._close_cache = close.sort_index()
        return self._close_cache

    def _get_benchmark_close(self) -> pd.Series | None:
        """提取基准收盘价序列（无基准时返回 ``None``）。"""
        if self._benchmark_close_cache is not None:
            return self._benchmark_close_cache
        if self.benchmark_data is None:
            return None

        bd = self.benchmark_data
        if self.benchmark_symbol is not None and isinstance(bd.index, pd.MultiIndex):
            level_vals = bd.index.get_level_values(Col.SYMBOL)
            key = self.benchmark_symbol
            if not (level_vals == self.benchmark_symbol).any():
                try:
                    key = level_vals.dtype.type(self.benchmark_symbol)
                except (ValueError, TypeError):
                    pass
            try:
                close = bd.xs(key, level=Col.SYMBOL)[self.price_col]
            except KeyError:
                logger.warning(
                    "基准 '{}' 数据未找到，超额相关指标将以 0 为基准。",
                    self.benchmark_symbol,
                )
                return None
        elif isinstance(bd.index, pd.MultiIndex):
            close = bd[self.price_col].groupby(level=Col.DATE).first()
        else:
            close = bd[self.price_col]

        close.index = pd.to_datetime(close.index)
        self._benchmark_close_cache = close.sort_index()
        return self._benchmark_close_cache

    def _get_returns(self) -> pd.Series:
        """目标股票逐日收益率（仅用于方向胜率的逐 bar 匹配）。"""
        return self._get_close().pct_change().rename("returns")

    def _get_benchmark_returns(self) -> pd.Series | None:
        """基准资产逐日收益率（用于超额收益计算）。"""
        bc = self._get_benchmark_close()
        if bc is None:
            return None
        return bc.pct_change().rename("benchmark_returns")

    # ------------------------------------------------------------------ #
    #  vectorbt 组合构建（私有，惰性求值）
    # ------------------------------------------------------------------ #

    def _portfolio(self):
        """构建策略 vectorbt Portfolio（惰性求值，结果缓存）。

        信号映射规则（T+1 交易逻辑）：

        - signal > 0 → 目标仓位 +100%（做多）
        - signal < 0 → 目标仓位 −100%（做空）
        - signal = 0 → 目标仓位   0%（空仓）

        信号在收盘后确认，次日生效（``shift(1)``）。
        """
        if self._pf is not None:
            return self._pf

        import vectorbt as vbt

        close = self._get_close()
        df = pd.DataFrame({"signal": self.factor_values, "close": close}).dropna()

        position = df["signal"].apply(
            lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
        )

        self._pf = vbt.Portfolio.from_orders(
            close=df["close"],
            size=position.shift(1).fillna(0.0),  # T+1 交易逻辑
            size_type="targetpercent",
            direction="both",                     # 允许做多 / 做空
            freq="D",
            init_cash=1.0,
            fees=0.0,
        )
        return self._pf

    def _benchmark_portfolio(self):
        """构建基准买入持有 vectorbt Portfolio（惰性求值，结果缓存）。"""
        if self._bm_pf is not None:
            return self._bm_pf

        bc = self._get_benchmark_close()
        if bc is None:
            return None

        import vectorbt as vbt

        self._bm_pf = vbt.Portfolio.from_orders(
            close=bc,
            size=1.0,
            size_type="targetpercent",
            direction="longonly",
            freq="D",
            init_cash=1.0,
            fees=0.0,
        )
        return self._bm_pf

    def _excess_portfolio(self):
        """将超额收益序列合成为价格曲线，再构建 vectorbt Portfolio。

        超额收益 = 策略日收益 − 基准日收益（无基准时等于策略日收益）。
        合成价格起点为 100，vbt 在首日自动买入并持有，后续净值变化等于
        超额累计收益，因此可以直接读取 vbt 的年化 / 回撤 / calmar 等指标。
        """
        if self._excess_pf is not None:
            return self._excess_pf

        import vectorbt as vbt

        excess = calc_excess_returns(
            self.strategy_returns(), self._get_benchmark_returns()
        ).dropna()
        if len(excess) < 2:
            return None

        # 在 excess 首日的前一天插入起始点 100，vbt 在此时建仓，
        # 之后净值变化正好等于 excess 的逐日复利累积。
        start_date = excess.index[0] - pd.Timedelta(days=1)
        synthetic = pd.concat(
            [
                pd.Series([100.0], index=[start_date]),
                (1.0 + excess).cumprod() * 100.0,
            ]
        )

        self._excess_pf = vbt.Portfolio.from_orders(
            close=synthetic,
            size=1.0,
            size_type="targetpercent",
            direction="longonly",
            freq="D",
            init_cash=100.0,
            fees=0.0,
        )
        return self._excess_pf

    # ------------------------------------------------------------------ #
    #  策略序列（公开）
    # ------------------------------------------------------------------ #

    def strategy_returns(self) -> pd.Series:
        """策略逐日收益序列（来自 vbt Portfolio.returns()）。"""
        return self._portfolio().returns().rename("strategy_returns")

    def nav(self) -> pd.Series:
        """策略净值曲线（起始净值 = 1.0，来自 vbt Portfolio.value()）。"""
        pf = self._portfolio()
        return (pf.value() / pf.init_cash).rename("nav")

    def benchmark_nav(self) -> pd.Series | None:
        """基准买入持有净值曲线；无基准时返回 ``None``。"""
        bm = self._benchmark_portfolio()
        if bm is None:
            return None
        return (bm.value() / bm.init_cash).rename("benchmark_nav")

    # ------------------------------------------------------------------ #
    #  单项指标（公开）
    # ------------------------------------------------------------------ #

    def direction_win_rate(self) -> float:
        """方向胜率（信号方向与实际日收益方向一致的比例）。

        Notes
        -----
        vectorbt 的胜率基于「入场 → 出场」交易轮次，无法按逐 bar
        比较信号方向与当日涨跌，故此指标保留自定义实现。
        """
        return calc_direction_win_rate(self.factor_values, self._get_returns())

    def annual_return(self) -> float:
        """策略绝对年化收益率（vbt Portfolio.annualized_return）。"""
        return float(
            self._portfolio().annualized_return(year_freq=_year_freq(self.trading_days))
        )

    def max_drawdown(self) -> float:
        """策略最大回撤（vbt Portfolio.max_drawdown，返回正值）。"""
        return float(abs(self._portfolio().max_drawdown()))

    def sharpe(self) -> float:
        """策略年化夏普比率（vbt Portfolio.sharpe_ratio）。

        Notes
        -----
        vbt 的 ``risk_free`` 参数为**每期**（日）无风险收益率，
        此处将年化无风险利率除以 ``trading_days`` 转换后传入。
        """
        daily_rf = self.risk_free / self.trading_days
        return float(
            self._portfolio().sharpe_ratio(
                risk_free=daily_rf,
                year_freq=_year_freq(self.trading_days),
            )
        )

    def calmar(self) -> float:
        """绝对卡玛比率（vbt Portfolio.calmar_ratio）。"""
        return float(
            self._portfolio().calmar_ratio(year_freq=_year_freq(self.trading_days))
        )

    def excess_annual_return(self) -> float:
        """超额年化收益率（超额合成净值的 vbt annualized_return）。

        无基准时超额 = 绝对收益，等同于 ``annual_return()``。
        """
        epf = self._excess_portfolio()
        if epf is None:
            return float("nan")
        return float(epf.annualized_return(year_freq=_year_freq(self.trading_days)))

    def excess_calmar(self) -> float:
        """超额卡玛比率（超额合成净值的 vbt calmar_ratio）。"""
        epf = self._excess_portfolio()
        if epf is None:
            return float("nan")
        return float(epf.calmar_ratio(year_freq=_year_freq(self.trading_days)))

    # ------------------------------------------------------------------ #
    #  汇总报告
    # ------------------------------------------------------------------ #

    def summary(self) -> pd.DataFrame:
        """生成汇总指标 DataFrame，行索引为股票代码。

        Returns
        -------
        pd.DataFrame，包含以下列：

        - ``direction_win_rate``   : 方向胜率
        - ``annual_return``        : 绝对年化收益率
        - ``max_drawdown``         : 最大回撤
        - ``sharpe``               : 夏普比率
        - ``calmar``               : 绝对卡玛比率
        - ``excess_annual_return`` : 超额年化收益率
        - ``excess_calmar``        : 超额卡玛比率
        """
        if self._summary is not None:
            return self._summary

        self._summary = pd.DataFrame(
            [
                {
                    "symbol": self.symbol,
                    "direction_win_rate": round(self.direction_win_rate(), 4),
                    "annual_return": round(self.annual_return(), 4),
                    "max_drawdown": round(self.max_drawdown(), 4),
                    "sharpe": round(self.sharpe(), 4),
                    "calmar": round(self.calmar(), 4),
                    "excess_annual_return": round(self.excess_annual_return(), 4),
                    "excess_calmar": round(self.excess_calmar(), 4),
                }
            ]
        ).set_index("symbol")

        return self._summary

    def to_dict(self) -> dict:
        """返回汇总指标字典，格式为 ``{symbol: {metric: value}}``。"""
        return self.summary().to_dict(orient="index")

    def print(self) -> None:
        """在终端打印格式化的评估报告。"""
        factor_name = getattr(self.factor_values, "name", "unknown")
        logger.info("=" * 60)
        logger.info("择时因子评估报告: {} | 股票: {}", factor_name, self.symbol)
        logger.info("=" * 60)
        print(self.summary().to_string())
        logger.info("-" * 60)
