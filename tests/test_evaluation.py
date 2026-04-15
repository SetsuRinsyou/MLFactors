"""evaluation 层单元测试 — IC/ICIR/分层回测/报告。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.schema import Col
from evaluation.selection.ic import (
    calc_ic,
    calc_ic_series,
    calc_icir,
    calc_ic_decay,
    calc_turnover,
    calc_t_stat,
    calc_forward_returns,
)
from evaluation.selection.layered import layered_backtest


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def make_panel(n_dates: int = 30, n_symbols: int = 20, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """生成随机因子值和前向收益 (MultiIndex)。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    symbols = [f"S{i:03d}" for i in range(n_symbols)]

    idx = pd.MultiIndex.from_product([dates, symbols], names=[Col.DATE, Col.SYMBOL])
    factor = pd.Series(rng.standard_normal(len(idx)), index=idx, name="factor")
    returns = pd.Series(rng.standard_normal(len(idx)) * 0.02, index=idx, name="returns")
    return factor, returns


def make_market_df(n_dates: int = 60, n_symbols: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    records = []
    for d in dates:
        for s in symbols:
            price = float(rng.uniform(9, 11))
            records.append({Col.DATE: d, Col.SYMBOL: s,
                             Col.OPEN: price, Col.HIGH: price * 1.01,
                             Col.LOW: price * 0.99, Col.CLOSE: price,
                             Col.VOLUME: float(rng.integers(1000, 5000))})
    df = pd.DataFrame(records)
    return df.set_index([Col.DATE, Col.SYMBOL]).sort_index()


# ── calc_ic ────────────────────────────────────────────────────────────────────

class TestCalcIC:
    def test_rank_ic_in_range(self):
        rng = np.random.default_rng(42)
        factor = pd.Series(rng.standard_normal(100))
        returns = pd.Series(rng.standard_normal(100))
        ic = calc_ic(factor, returns, method="rank")
        assert -1.0 <= ic <= 1.0

    def test_pearson_ic_in_range(self):
        rng = np.random.default_rng(42)
        factor = pd.Series(rng.standard_normal(100))
        returns = pd.Series(rng.standard_normal(100))
        ic = calc_ic(factor, returns, method="pearson")
        assert -1.0 <= ic <= 1.0

    def test_perfect_positive_correlation(self):
        x = pd.Series(range(50), dtype=float)
        ic = calc_ic(x, x, method="rank")
        assert abs(ic - 1.0) < 1e-9

    def test_perfect_negative_correlation(self):
        x = pd.Series(range(50), dtype=float)
        y = -x
        ic = calc_ic(x, y, method="rank")
        assert abs(ic + 1.0) < 1e-9

    def test_too_few_samples_returns_nan(self):
        factor = pd.Series([1.0, 2.0])
        returns = pd.Series([1.0, 2.0])
        ic = calc_ic(factor, returns)
        assert np.isnan(ic)

    def test_handles_nan_values(self):
        factor = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ic = calc_ic(factor, returns)
        assert not np.isnan(ic)


# ── calc_ic_series ─────────────────────────────────────────────────────────────

class TestCalcICSeries:
    def test_output_is_series(self):
        factor, returns = make_panel()
        result = calc_ic_series(factor, returns)
        assert isinstance(result, pd.Series)

    def test_index_is_dates(self):
        factor, returns = make_panel(n_dates=10)
        result = calc_ic_series(factor, returns)
        assert len(result) <= 10

    def test_values_in_range(self):
        factor, returns = make_panel()
        result = calc_ic_series(factor, returns)
        valid = result.dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_accepts_dataframe_input(self):
        factor, returns = make_panel()
        result = calc_ic_series(factor.to_frame(), returns.to_frame())
        assert isinstance(result, pd.Series)

    def test_positive_factor_has_higher_mean_ic(self):
        """完全正相关因子应产生正平均 IC。"""
        rng = np.random.default_rng(7)
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        symbols = [f"S{i:03d}" for i in range(30)]
        idx = pd.MultiIndex.from_product([dates, symbols], names=[Col.DATE, Col.SYMBOL])
        factor = pd.Series(rng.standard_normal(len(idx)), index=idx)
        # 收益 = 因子 + 噪声
        returns = factor + pd.Series(rng.standard_normal(len(idx)) * 0.1, index=idx)
        ic_s = calc_ic_series(factor, returns, method="rank")
        assert ic_s.mean() > 0.5


# ── calc_icir ─────────────────────────────────────────────────────────────────

class TestCalcICIR:
    def test_icir_formula(self):
        ic_series = pd.Series([0.1, 0.2, 0.15, 0.05, 0.1])
        expected = ic_series.mean() / ic_series.std()
        assert abs(calc_icir(ic_series) - expected) < 1e-10

    def test_zero_std_not_crash(self):
        """常数 IC 序列的 std 极小，不应抛异常，结果为有限大数或 inf。"""
        ic_series = pd.Series([0.1, 0.1, 0.1])
        result = calc_icir(ic_series)
        assert isinstance(result, float)

    def test_too_short_returns_nan(self):
        assert np.isnan(calc_icir(pd.Series([0.1])))

    def test_positive_signal_positive_icir(self):
        factor, returns = make_panel(n_dates=30, n_symbols=50, seed=7)
        # 让 factor 和 returns 正相关
        returns_pos = factor + pd.Series(
            np.random.default_rng(9).standard_normal(len(factor)) * 0.1,
            index=factor.index,
        )
        ic_s = calc_ic_series(factor, returns_pos)
        assert calc_icir(ic_s) > 0


# ── calc_ic_decay ─────────────────────────────────────────────────────────────

class TestCalcICDecay:
    def test_decay_length(self):
        factor, returns = make_panel(n_dates=60, n_symbols=20)
        decay = calc_ic_decay(factor, returns, max_lag=5)
        assert len(decay) == 5

    def test_decay_index(self):
        factor, returns = make_panel(n_dates=60)
        decay = calc_ic_decay(factor, returns, max_lag=5)
        assert list(decay.index) == [1, 2, 3, 4, 5]

    def test_decay_values_in_range(self):
        factor, returns = make_panel(n_dates=60)
        decay = calc_ic_decay(factor, returns, max_lag=5)
        assert (decay.dropna().abs() <= 1.0).all()


# ── calc_turnover ─────────────────────────────────────────────────────────────

class TestCalcTurnover:
    def test_turnover_in_range(self):
        factor, _ = make_panel(n_dates=20, n_symbols=10)
        result = calc_turnover(factor)
        assert (result >= 0).all()
        assert (result <= 1.0).all()

    def test_static_factor_zero_turnover(self):
        """不随时间变化的因子换手率应为零。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        symbols = ["A", "B", "C"]
        idx = pd.MultiIndex.from_product([dates, symbols], names=[Col.DATE, Col.SYMBOL])
        # 同一时序结构，每只股票跨时间排名不变
        vals = [1.0, 2.0, 3.0] * 10
        factor = pd.Series(vals, index=idx)
        turnover = calc_turnover(factor)
        assert (turnover.dropna() < 1e-10).all()


# ── calc_t_stat ───────────────────────────────────────────────────────────────

class TestCalcTStat:
    def test_returns_two_values(self):
        ic = pd.Series([0.1, 0.05, 0.08, 0.12, 0.09])
        t, p = calc_t_stat(ic)
        assert isinstance(t, float)
        assert isinstance(p, float)

    def test_p_in_range(self):
        ic = pd.Series(np.random.default_rng(0).standard_normal(50) * 0.05 + 0.03)
        _, p = calc_t_stat(ic)
        assert 0.0 <= p <= 1.0

    def test_significant_ic_low_pvalue(self):
        ic = pd.Series([0.15] * 100)  # 常数序列 std=0，会退化
        # 用有随机性的强信号
        ic = pd.Series(np.random.default_rng(1).normal(0.1, 0.03, 60))
        _, p = calc_t_stat(ic)
        assert p < 0.05

    def test_too_short_returns_nan(self):
        t, p = calc_t_stat(pd.Series([0.1]))
        assert np.isnan(t) and np.isnan(p)


# ── calc_forward_returns ──────────────────────────────────────────────────────

class TestCalcForwardReturns:
    def test_returns_dict_of_series(self):
        mkt = make_market_df()
        result = calc_forward_returns(mkt, periods=[1, 5])
        assert set(result.keys()) == {1, 5}
        for s in result.values():
            assert isinstance(s, pd.Series)

    def test_series_name(self):
        mkt = make_market_df()
        result = calc_forward_returns(mkt, periods=[5])
        assert result[5].name == "fwd_ret_5"

    def test_positive_trend_positive_forward_return(self):
        """单调上涨资产的前向收益应为正。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        idx = pd.MultiIndex.from_product([dates, ["A"]], names=[Col.DATE, Col.SYMBOL])
        prices = list(range(1, 31))
        mkt = pd.DataFrame({Col.CLOSE: prices, Col.OPEN: prices,
                             Col.HIGH: prices, Col.LOW: prices,
                             Col.VOLUME: [100] * 30}, index=idx)
        result = calc_forward_returns(mkt, periods=[1])
        valid = result[1].dropna()
        assert (valid > 0).all()


# ── layered_backtest ──────────────────────────────────────────────────────────

class TestLayeredBacktest:
    def test_returns_layered_result(self):
        factor, returns = make_panel(n_dates=40, n_symbols=30)
        from evaluation.selection.layered import LayeredResult
        result = layered_backtest(factor, returns, n_groups=5)
        assert isinstance(result, LayeredResult)

    def test_n_groups_columns(self):
        factor, returns = make_panel(n_dates=40, n_symbols=30)
        result = layered_backtest(factor, returns, n_groups=5)
        assert len(result.group_returns.columns) == 5

    def test_annual_returns_shape(self):
        factor, returns = make_panel(n_dates=40, n_symbols=30)
        result = layered_backtest(factor, returns, n_groups=5)
        assert len(result.annual_returns) == 5

    def test_sharpe_ratios_shape(self):
        factor, returns = make_panel(n_dates=40, n_symbols=30)
        result = layered_backtest(factor, returns, n_groups=5)
        assert len(result.sharpe_ratios) == 5

    def test_long_short_returns_length(self):
        factor, returns = make_panel(n_dates=40, n_symbols=30)
        result = layered_backtest(factor, returns, n_groups=5)
        assert len(result.long_short_returns) > 0

    def test_cumulative_starts_near_zero(self):
        factor, returns = make_panel(n_dates=40, n_symbols=30)
        result = layered_backtest(factor, returns, n_groups=5)
        first_row = result.cumulative_returns.iloc[0]
        assert (first_row.abs() < 0.5).all()

    def test_positive_factor_spread(self):
        """强正相关因子的最高组应跑赢最低组。"""
        rng = np.random.default_rng(5)
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        symbols = [f"S{i:03d}" for i in range(50)]
        idx = pd.MultiIndex.from_product([dates, symbols], names=[Col.DATE, Col.SYMBOL])
        factor = pd.Series(rng.standard_normal(len(idx)), index=idx)
        returns = factor * 0.01 + pd.Series(rng.standard_normal(len(idx)) * 0.001, index=idx)
        result = layered_backtest(factor, returns, n_groups=5)
        assert result.annual_returns[5] > result.annual_returns[1]

    def test_accepts_dataframe_inputs(self):
        factor, returns = make_panel(n_dates=40, n_symbols=30)
        result = layered_backtest(factor.to_frame(), returns.to_frame(), n_groups=3)
        assert result.n_groups == 3


# ── FactorReport 集成测试 ─────────────────────────────────────────────────────

class TestFactorReport:
    def test_summary_returns_dataframe(self):
        from evaluation.selection.report import FactorReport
        from factors.registry import FactorRegistry
        FactorRegistry.reset()

        mkt = make_market_df(n_dates=60, n_symbols=20)
        cls = FactorRegistry.get("momentum_5")
        factor_vals = cls().compute(mkt).dropna()

        report = FactorReport(
            factor_values=factor_vals,
            market_data=mkt,
            forward_periods=[1, 5],
            n_groups=5,
        )
        summary = report.summary()
        assert isinstance(summary, pd.DataFrame)
        assert set(summary.index) == {1, 5}
        assert "IC_mean" in summary.columns
        assert "ICIR" in summary.columns

    def test_to_dict(self):
        from evaluation.selection.report import FactorReport
        from factors.registry import FactorRegistry
        FactorRegistry.reset()

        mkt = make_market_df(n_dates=60, n_symbols=20)
        cls = FactorRegistry.get("momentum_5")
        factor_vals = cls().compute(mkt).dropna()

        report = FactorReport(factor_values=factor_vals, market_data=mkt,
                              forward_periods=[1], n_groups=5)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert 1 in d
