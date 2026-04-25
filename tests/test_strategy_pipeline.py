from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.execution import SimpleBacktestExecutionEngine
from backtest.portfolio import SimpleTopKPortfolioManager
from data.schema import Col
from evaluation.strategy_analyzer import SimpleStrategyAnalyzer
from factors.library.selection.momentum import Momentum5, Momentum20
from pipeline.strategy_runner import StrategyPipeline


def make_market_data(
    start: str = "2020-01-01",
    end: str = "2026-01-01",
    symbols: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    if symbols is None:
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    records: list[dict[str, float | str | pd.Timestamp]] = []

    for idx, symbol in enumerate(symbols):
        base = 100 + idx * 10
        drift = 0.0003 + idx * 0.00005
        vol = 0.012 + idx * 0.001
        returns = rng.normal(drift, vol, len(dates))
        close = base * np.exp(np.cumsum(returns))
        open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, len(dates))))
        volume = rng.integers(1_000_000, 5_000_000, len(dates)).astype(float)

        for date, op, hi, lo, cl, vol_ in zip(dates, open_, high, low, close, volume, strict=True):
            records.append(
                {
                    Col.DATE: date,
                    Col.SYMBOL: symbol,
                    Col.OPEN: float(op),
                    Col.HIGH: float(hi),
                    Col.LOW: float(lo),
                    Col.CLOSE: float(cl),
                    Col.ADJ_CLOSE: float(cl),
                    Col.VOLUME: float(vol_),
                }
            )

    return pd.DataFrame(records).set_index([Col.DATE, Col.SYMBOL]).sort_index()


def test_momentum5_generates_panel():
    market = make_market_data(end="2020-03-31")
    model = Momentum5()

    signals = model.generate_signals(market)

    assert isinstance(signals, pd.DataFrame)
    assert list(signals.columns) == ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
    assert signals.iloc[:5].isna().all().all()
    assert signals.iloc[6:].notna().any().any()


def test_simple_topk_portfolio_manager_outputs_fully_invested_weights():
    market = make_market_data(end="2020-03-31")
    signals = Momentum5().generate_signals(market)
    manager = SimpleTopKPortfolioManager(top_k=2, rebalance_frequency="W-FRI")

    weights = manager.generate_target_weights(signals, market)
    row_sums = weights.sum(axis=1)

    assert ((row_sums == 0) | np.isclose(row_sums, 1.0)).all()
    assert (weights[weights > 0].count(axis=1) <= 2).all()
    assert weights.loc["2020-01-13":"2020-01-16"].nunique().max() == 1


def test_simple_execution_engine_applies_shift_one():
    dates = pd.bdate_range("2020-01-01", periods=4)
    symbols = ["AAPL"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=[Col.DATE, Col.SYMBOL])
    market = pd.DataFrame(
        {
            Col.CLOSE: [100.0, 110.0, 121.0, 133.1],
            Col.OPEN: [100.0, 110.0, 121.0, 133.1],
            Col.HIGH: [100.0, 110.0, 121.0, 133.1],
            Col.LOW: [100.0, 110.0, 121.0, 133.1],
            Col.VOLUME: [1_000_000.0] * 4,
        },
        index=idx,
    )
    target_weights = pd.DataFrame({"AAPL": [1.0, 1.0, 1.0, 1.0]}, index=dates)
    target_weights.index.name = Col.DATE

    result = SimpleBacktestExecutionEngine(
        commission=0.0,
        slippage=0.0,
        initial_capital=1_000.0,
    ).run_simulation(
        target_weights=target_weights,
        market_data=market,
    )

    assert result.positions.iloc[0, 0] == 0.0
    assert result.positions.iloc[1, 0] == 9.0
    assert np.isclose(result.returns.iloc[0], 0.0)
    assert np.isclose(result.returns.iloc[1], 0.0)
    assert np.isclose(result.returns.iloc[2], 0.099)
    assert float(result.meta["initial_capital"]) == 1000.0
    trade_details = result.meta["trade_details"]
    assert list(trade_details[[Col.SYMBOL, "action", "shares", "price"]].iloc[0]) == [
        "AAPL",
        "买入",
        9,
        110.0,
    ]


def test_strategy_pipeline_runs_end_to_end():
    market = make_market_data()
    close = market[Col.CLOSE].unstack(Col.SYMBOL).sort_index()
    benchmark = pd.DataFrame(
        {
            "SPY": close.mean(axis=1).pct_change().fillna(0.0),
            "QQQ": close.iloc[:, :3].mean(axis=1).pct_change().fillna(0.0),
        }
    )

    pipeline = StrategyPipeline(
        alpha_model=Momentum20(),
        portfolio_manager=SimpleTopKPortfolioManager(top_k=3, rebalance_frequency="W-FRI"),
        execution_engine=SimpleBacktestExecutionEngine(
            commission=0.0005,
            slippage=0.0005,
            initial_capital=1_000_000.0,
        ),
        analyzer_cls=SimpleStrategyAnalyzer,
    )
    result = pipeline.run(market_data=market, benchmark_returns=benchmark)

    assert set(result.keys()) == {"signals", "target_weights", "simulation", "stats", "analyzer"}
    assert len(result["simulation"].equity_curve) > 200
    assert np.issubdtype(result["simulation"].positions.dtypes.iloc[0], np.integer)
    assert "annual_return" in result["stats"]
    assert "win_rate" in result["stats"]
    assert "information_ratio_SPY" in result["stats"]
    assert "information_ratio_QQQ" in result["stats"]
    assert result["stats"]["initial_capital"] == 1_000_000.0
    assert np.isfinite(result["stats"]["max_drawdown"])
    assert 0.0 <= result["stats"]["win_rate"] <= 1.0


def test_strategy_analyzer_exports_vectorbt_report(tmp_path):
    market = make_market_data(end="2020-06-30")
    close = market[Col.CLOSE].unstack(Col.SYMBOL).sort_index()
    benchmark = pd.DataFrame(
        {
            "SPY": close.mean(axis=1).pct_change().fillna(0.0),
            "QQQ": close.iloc[:, :3].mean(axis=1).pct_change().fillna(0.0),
        }
    )

    pipeline = StrategyPipeline(
        alpha_model=Momentum5(),
        portfolio_manager=SimpleTopKPortfolioManager(top_k=2, rebalance_frequency="W-FRI"),
        execution_engine=SimpleBacktestExecutionEngine(
            commission=0.0005,
            slippage=0.0005,
            initial_capital=500_000.0,
        ),
        analyzer_cls=SimpleStrategyAnalyzer,
    )
    result = pipeline.run(market_data=market, benchmark_returns=benchmark)
    expected_records = (
        result["simulation"].meta["trade_details"]
        .rename(columns={"cash_after": "balance"})
        [[Col.DATE, Col.SYMBOL, "action", "shares", "price", "notional", "balance"]]
        .reset_index(drop=True)
    )
    actual_records = result["analyzer"].rebalance_records()
    pd.testing.assert_frame_equal(
        actual_records.head(10),
        expected_records.head(10),
        check_dtype=False,
    )

    report_path = result["analyzer"].export_report(tmp_path / "report")

    assert report_path.exists()
    assert report_path.name == "report.html"
    assert (tmp_path / "report" / "summary_stats.csv").exists()
    assert (tmp_path / "report" / "vectorbt_stats.csv").exists()
    assert (tmp_path / "report" / "rebalance_records.csv").exists()
    report_html = report_path.read_text(encoding="utf-8")
    assert "Rebalance Records" in report_html
    assert "Win Rate" in report_html
    assert "metric-card" in report_html
    assert "date-band-even" in report_html
    assert "date-band-odd" in report_html
    stats = pd.read_csv(tmp_path / "report" / "summary_stats.csv", index_col=0)
    assert "win_rate" in stats.index
    records = pd.read_csv(tmp_path / "report" / "rebalance_records.csv")
    assert list(records.columns) == [
        Col.DATE,
        Col.SYMBOL,
        "action",
        "shares",
        "price",
        "notional",
        "balance",
    ]
