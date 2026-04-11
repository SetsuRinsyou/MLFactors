"""factors 层单元测试 — BaseFactor, FactorRegistry, 内置因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.schema import Col
from factors.base import BaseFactor
from factors.registry import FactorRegistry, register_factor


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def make_market_df(n_dates: int = 30, n_symbols: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    rng = np.random.default_rng(42)
    records = []
    for d in dates:
        for s in symbols:
            price = float(rng.uniform(9, 11))
            records.append({
                Col.DATE: d,
                Col.SYMBOL: s,
                Col.OPEN: price,
                Col.HIGH: price * 1.02,
                Col.LOW: price * 0.98,
                Col.CLOSE: price,
                Col.VOLUME: float(rng.integers(1000, 5000)),
            })
    df = pd.DataFrame(records)
    return df.set_index([Col.DATE, Col.SYMBOL]).sort_index()


# ── BaseFactor 接口测试 ───────────────────────────────────────────────────────

class TestBaseFactor:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseFactor()

    def test_subclass_requires_compute(self):
        class Incomplete(BaseFactor):
            name = "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_ok(self):
        class SimpleFactor(BaseFactor):
            name = "simple"

            def compute(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE].rename(self.name)

        f = SimpleFactor()
        assert f.name == "simple"

    def test_repr(self):
        class SimpleFactor(BaseFactor):
            name = "simple"
            category = "test"

            def compute(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE]

        assert "simple" in repr(SimpleFactor())


# ── FactorRegistry 测试 ───────────────────────────────────────────────────────

class TestFactorRegistry:
    def setup_method(self):
        # 保存并隔离注册表状态，测试后还原
        self._backup = dict(FactorRegistry._registry)
        self._loaded_backup = FactorRegistry._loaded
        FactorRegistry._registry = {}
        FactorRegistry._loaded = True  # 阻止自动发现以便隔离测试

    def teardown_method(self):
        FactorRegistry._registry = self._backup
        FactorRegistry._loaded = self._loaded_backup

    def test_register_and_get(self):
        @register_factor
        class Dummy(BaseFactor):
            name = "dummy_test"

            def compute(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE]

        assert FactorRegistry.get("dummy_test") is Dummy

    def test_list_sorted(self):
        @register_factor
        class B(BaseFactor):
            name = "b_factor"

            def compute(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE]

        @register_factor
        class A(BaseFactor):
            name = "a_factor"

            def compute(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE]

        names = FactorRegistry.list()
        assert names == sorted(names)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            FactorRegistry.get("nonexistent_xyz")

    def test_register_empty_name_raises(self):
        with pytest.raises(ValueError):
            @register_factor
            class NoName(BaseFactor):
                name = ""

                def compute(self, market_data, fundamental_data=None):
                    return market_data[Col.CLOSE]

    def test_compute_all_returns_dataframe(self):
        @register_factor
        class F1(BaseFactor):
            name = "f1_test"

            def compute(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE].rename(self.name)

        @register_factor
        class F2(BaseFactor):
            name = "f2_test"

            def compute(self, market_data, fundamental_data=None):
                return (market_data[Col.CLOSE] * 2).rename(self.name)

        mkt = make_market_df()
        result = FactorRegistry.compute_all(mkt, factor_names=["f1_test", "f2_test"])
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"f1_test", "f2_test"}

    def test_list_detail_has_fields(self):
        @register_factor
        class Detail(BaseFactor):
            name = "detail_test"
            description = "test desc"
            category = "test_cat"

            def compute(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE]

        details = FactorRegistry.list_detail()
        entry = next(d for d in details if d["name"] == "detail_test")
        assert entry["description"] == "test desc"
        assert entry["category"] == "test_cat"


# ── 内置因子测试 ──────────────────────────────────────────────────────────────

class TestBuiltinFactors:
    """验证内置因子可以正确自动注册并计算。"""

    def setup_method(self):
        FactorRegistry.reset()

    def teardown_method(self):
        FactorRegistry.reset()

    def _mkt(self):
        return make_market_df(n_dates=40, n_symbols=5)

    def test_builtin_factors_auto_registered(self):
        names = FactorRegistry.list()
        assert "momentum_5" in names
        assert "volatility_20" in names
        assert "highlow_spread_20" in names

    def test_momentum5_output_shape(self):
        mkt = self._mkt()
        cls = FactorRegistry.get("momentum_5")
        result = cls().compute(mkt)
        assert isinstance(result, pd.Series)
        # 因 pct_change(5) 前5行为 NaN，应有非空值
        assert result.dropna().__len__() > 0

    def test_momentum5_output_index(self):
        mkt = self._mkt()
        cls = FactorRegistry.get("momentum_5")
        result = cls().compute(mkt)
        assert result.index.names == [Col.DATE, Col.SYMBOL]

    def test_volatility20_nonnegative(self):
        mkt = self._mkt()
        cls = FactorRegistry.get("volatility_20")
        result = cls().compute(mkt).dropna()
        assert (result >= 0).all()

    def test_highlow_spread_nonnegative(self):
        mkt = self._mkt()
        cls = FactorRegistry.get("highlow_spread_20")
        result = cls().compute(mkt).dropna()
        assert (result >= 0).all()
