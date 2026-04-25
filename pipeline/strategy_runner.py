"""策略回测总调度流水线 — 串联 5 大核心模块，执行端到端回测。"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from backtest.execution import BaseExecutionEngine, SimulationResult
from backtest.portfolio import BasePortfolioManager
from evaluation.strategy_analyzer import BaseStrategyAnalyzer
from factors import BaseFactor


class StrategyPipeline:
    """策略回测总调度流水线 — 串联 4 大核心模块，执行端到端回测。

    通过依赖注入接收 4 个核心组件实例，按以下顺序单向透传数据::

        market_data
            │
            ▼
        ┌──────────────────────┐
        │    BaseFactor        │  → signals
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ BasePortfolioManager │  → target_weights
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ BaseExecutionEngine  │  → SimulationResult
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ BaseStrategyAnalyzer │  → stats + tearsheet
        └──────────────────────┘

    Parameters
    ----------
    alpha_model : BaseFactor
        选股因子或信号引擎实例。
    portfolio_manager : BasePortfolioManager
        仓位与风控管理实例。
    execution_engine : BaseExecutionEngine
        撮合引擎实例（已配置佣金与滑点）。
    analyzer_cls : type[BaseStrategyAnalyzer]
        绩效分析**类**（非实例），Pipeline 在撮合完成后自动实例化。
    """

    def __init__(
        self,
        alpha_model: BaseFactor,
        portfolio_manager: BasePortfolioManager,
        execution_engine: BaseExecutionEngine,
        analyzer_cls: type[BaseStrategyAnalyzer],
    ) -> None:
        self.alpha_model = alpha_model
        self.portfolio_manager = portfolio_manager
        self.execution_engine = execution_engine
        self.analyzer_cls = analyzer_cls

    def run(
        self,
        market_data: pd.DataFrame,
        benchmark_returns: pd.Series | None = None,
    ) -> dict[str, Any]:
        """执行完整策略回测流水线。

        Parameters
        ----------
        market_data : pd.DataFrame
            已清洗的标准面板行情数据，MultiIndex ``(date, symbol)``。
            上游数据层已完成除权复权、停牌处理、退市股过滤等清洗工作。
        benchmark_returns : pd.Series | None
            基准收益率序列（日频），索引为日期。若提供则计算超额收益指标。

        Returns
        -------
        dict[str, Any]
            包含以下键值：

            - ``"signals"``        : pd.DataFrame — 原始信号矩阵
            - ``"target_weights"`` : pd.DataFrame — 目标权重矩阵
            - ``"simulation"``     : SimulationResult — 撮合结果
            - ``"stats"``          : dict[str, float] — 绩效统计
            - ``"analyzer"``       : BaseStrategyAnalyzer — 分析器实例（可调用 plot_tearsheet）
        """
        # ── 阶段 1：信号生成 ──────────────────────────────────────────
        logger.info("▶ [1/4] 信号引擎：生成 Alpha 信号 ...")
        signals = self.alpha_model.generate_signals(market_data)
        logger.info(
            "  信号矩阵: {} 个交易日 × {} 个标的",
            signals.shape[0], signals.shape[1],
        )

        # ── 阶段 2：仓位与风控 ────────────────────────────────────────
        logger.info("▶ [2/4] 仓位管理：生成目标权重 ...")
        target_weights = self.portfolio_manager.generate_target_weights(
            signals, market_data,
        )
        n_active = (target_weights != 0).sum(axis=1).mean()
        logger.info(
            "  目标权重矩阵: {} 个交易日 × {} 个标的, 平均持仓 {:.1f} 只",
            target_weights.shape[0], target_weights.shape[1], n_active,
        )

        # ── 阶段 3：撮合执行 ──────────────────────────────────────────
        logger.info(
            "▶ [3/4] 撮合引擎：执行回测 (佣金={:.4f}, 滑点={:.4f}) ...",
            self.execution_engine.commission,
            self.execution_engine.slippage,
        )
        simulation = self.execution_engine.run_simulation(
            target_weights, market_data,
        )
        logger.info(
            "  回测完成: {} 个交易日, 终值净值 {:.4f}",
            len(simulation.equity_curve),
            simulation.equity_curve.iloc[-1] if len(simulation.equity_curve) > 0 else 0,
        )

        # ── 阶段 4：绩效分析 ──────────────────────────────────────────
        logger.info("▶ [4/4] 绩效分析：计算统计指标 ...")
        try:
            analyzer = self.analyzer_cls(
                simulation,
                benchmark_returns,
                market_data=market_data,
            )
        except TypeError:
            analyzer = self.analyzer_cls(simulation, benchmark_returns)
        stats = analyzer.summary_stats()
        logger.info(
            "  年化收益={:.2%}  夏普={:.2f}  最大回撤={:.2%}  卡玛={:.2f}",
            stats.get("annual_return", 0),
            stats.get("sharpe_ratio", 0),
            stats.get("max_drawdown", 0),
            stats.get("calmar_ratio", 0),
        )

        logger.info("✔ 策略回测流水线执行完毕")
        return {
            "signals": signals,
            "target_weights": target_weights,
            "simulation": simulation,
            "stats": stats,
            "analyzer": analyzer,
        }
