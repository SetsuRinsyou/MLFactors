"""美股策略回测示例脚本。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
from loguru import logger

from backtest import SimpleBacktestExecutionEngine, SimpleTopKPortfolioManager
from data.local_loader import USStockLocalLoader
from data.schema import Col
from evaluation import SimpleStrategyAnalyzer
from factors.base import BaseFactor
from factors.library.selection.momentum import Momentum5, Momentum10, Momentum20
from pipeline import StrategyPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行最小美股策略回测示例")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("/home/setsu/workspace/data/cta_orange.db"),
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"],
    )
    parser.add_argument("--benchmarks", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--lookback", type=int, choices=[5, 10, 20], default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/strategy"))
    return parser.parse_args()


def build_momentum_factor(lookback: int) -> BaseFactor:
    factors = {
        5: Momentum5,
        10: Momentum10,
        20: Momentum20,
    }
    return factors[lookback]()


def load_market_data(db_path: Path, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"未找到美股 SQLite 数据库: {db_path}")

    logger.info("使用本地 SQLite 美股数据: {}", db_path)
    loader = USStockLocalLoader(db_path=db_path)
    market_data = loader.load_market_data(symbols=symbols, start=start, end=end)
    if market_data.empty:
        raise RuntimeError(
            f"SQLite 中未查到指定区间/标的数据: symbols={symbols}, start={start}, end={end}"
    )
    return market_data


def split_strategy_and_benchmark_data(
    market_data: pd.DataFrame,
    strategy_symbols: list[str],
    benchmark_symbols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol_index = market_data.index.get_level_values(Col.SYMBOL)
    strategy_data = market_data.loc[symbol_index.isin(strategy_symbols)]
    benchmark_data = market_data.loc[symbol_index.isin(benchmark_symbols)]
    return strategy_data, benchmark_data


def build_benchmark_returns(
    benchmark_data: pd.DataFrame,
    benchmark_symbols: list[str],
) -> pd.DataFrame:
    prices = benchmark_data[(Col.ADJ_CLOSE if Col.ADJ_CLOSE in benchmark_data.columns else Col.CLOSE)]
    close = prices.unstack(Col.SYMBOL).sort_index()
    returns = close.pct_change().fillna(0.0)
    benchmark_returns = returns.reindex(columns=benchmark_symbols).dropna(how="all")
    return benchmark_returns


def save_outputs(result: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    simulation = result["simulation"]
    analyzer = result.get("analyzer")
    stats = pd.Series(result["stats"], name="value")
    stats.to_csv(output_dir / "stats.csv", header=True)
    simulation.equity_curve.to_csv(output_dir / "equity_curve.csv", header=True)
    simulation.returns.to_csv(output_dir / "returns.csv", header=True)
    simulation.positions.to_csv(output_dir / "positions.csv")
    if isinstance(simulation.meta.get("position_weights"), pd.DataFrame):
        simulation.meta["position_weights"].to_csv(output_dir / "position_weights.csv")
    if isinstance(simulation.meta.get("portfolio_value"), pd.Series):
        simulation.meta["portfolio_value"].to_csv(output_dir / "portfolio_value.csv", header=True)
    if isinstance(simulation.meta.get("cash_curve"), pd.Series):
        simulation.meta["cash_curve"].to_csv(output_dir / "cash_curve.csv", header=True)
    if simulation.trades is not None:
        simulation.trades.to_csv(output_dir / "trades.csv")
    if isinstance(simulation.meta.get("trade_details"), pd.DataFrame):
        simulation.meta["trade_details"].to_csv(output_dir / "trade_details.csv", index=False)
    if analyzer is not None and hasattr(analyzer, "export_report"):
        report_path = analyzer.export_report(output_dir / "report")
        logger.info("策略 HTML 报告已保存: {}", report_path)
    logger.info("回测结果已保存到 {}", output_dir)


def main() -> None:
    args = parse_args()
    all_symbols = list(dict.fromkeys([*args.symbols, *args.benchmarks]))
    market_data = load_market_data(
        db_path=args.db_path,
        symbols=all_symbols,
        start=args.start,
        end=args.end,
    )
    if market_data.empty:
        raise RuntimeError("未能加载到任何行情数据，无法执行回测")

    strategy_data, benchmark_data = split_strategy_and_benchmark_data(
        market_data,
        strategy_symbols=args.symbols,
        benchmark_symbols=args.benchmarks,
    )
    benchmark_returns = build_benchmark_returns(benchmark_data, args.benchmarks)
    pipeline = StrategyPipeline(
        alpha_model=build_momentum_factor(args.lookback),
        portfolio_manager=SimpleTopKPortfolioManager(top_k=args.top_k, rebalance_frequency="W-FRI"),
        execution_engine=SimpleBacktestExecutionEngine(
            commission=0.0005,
            slippage=0.0005,
            initial_capital=args.initial_capital,
        ),
        analyzer_cls=SimpleStrategyAnalyzer,
    )
    result = pipeline.run(market_data=strategy_data, benchmark_returns=benchmark_returns)
    save_outputs(result, args.output_dir)

    stats = result["stats"]
    logger.info(
        "示例回测完成: initial_capital={:.0f}, ending_value={:.2f}, total_return={:.2%}, annual_return={:.2%}, win_rate={:.2%}, sharpe={:.2f}, max_drawdown={:.2%}",
        stats["initial_capital"],
        stats["ending_value"],
        stats["total_return"],
        stats["annual_return"],
        stats["win_rate"],
        stats["sharpe_ratio"],
        stats["max_drawdown"],
    )


if __name__ == "__main__":
    main()
