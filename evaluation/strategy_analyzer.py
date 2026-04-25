"""绩效分析器基类与基于 vectorbt 的策略报告实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger

from backtest.execution import SimulationResult
from data.schema import Col


def _year_freq(trading_days: int) -> str:
    return f"{trading_days}D"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(result):
        return default
    return result


def _ensure_vectorbt():
    import vectorbt as vbt

    return vbt


def _format_table_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:,.6f}"
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    return escape(str(value))


class BaseStrategyAnalyzer(ABC):
    """绩效分析器基类。"""

    def __init__(
        self,
        simulation_result: SimulationResult,
        benchmark_returns: pd.Series | pd.DataFrame | None = None,
        market_data: pd.DataFrame | None = None,
    ) -> None:
        self.result = simulation_result
        self.benchmark_returns = benchmark_returns
        self.market_data = market_data

    @abstractmethod
    def summary_stats(self) -> dict[str, float]:
        """输出核心绩效统计指标。"""
        ...

    @abstractmethod
    def plot_tearsheet(
        self,
        output_dir: str | Path | None = None,
        show: bool = False,
    ) -> None:
        """绘制或导出基础策略报告。"""
        ...

    def export_report(self, output_dir: str | Path) -> Path:
        """导出策略报告。"""
        raise NotImplementedError


class SimpleStrategyAnalyzer(BaseStrategyAnalyzer):
    """基于 vectorbt 的最小策略绩效分析器。"""

    def __init__(
        self,
        simulation_result: SimulationResult,
        benchmark_returns: pd.Series | pd.DataFrame | None = None,
        market_data: pd.DataFrame | None = None,
        trading_days: int = 252,
        risk_free: float = 0.0,
        price_col: str | None = None,
        strategy_name: str = "Strategy",
    ) -> None:
        super().__init__(
            simulation_result=simulation_result,
            benchmark_returns=benchmark_returns,
            market_data=market_data,
        )
        self.trading_days = trading_days
        self.risk_free = risk_free
        self.price_col = price_col
        self.strategy_name = strategy_name

        self._pf = None
        self._benchmark_pfs: dict[str, object] = {}
        self._stats_cache: dict[str, float] | None = None

    def _resolve_price_col(self) -> str:
        if self.price_col is not None:
            return self.price_col
        if self.market_data is not None and Col.ADJ_CLOSE in self.market_data.columns:
            return Col.ADJ_CLOSE
        return Col.CLOSE

    def _initial_capital(self) -> float:
        return _safe_float(self.result.meta.get("initial_capital", 1.0), default=1.0)

    def _strategy_returns(self) -> pd.Series:
        return self.result.returns.fillna(0.0).sort_index()

    def _strategy_nav(self) -> pd.Series:
        nav = self.result.equity_curve.ffill().fillna(1.0).sort_index()
        nav.name = "equity"
        return nav

    def _portfolio_value_curve(self) -> pd.Series:
        curve = self.result.meta.get("portfolio_value")
        if isinstance(curve, pd.Series):
            return curve.sort_index()
        return (self._strategy_nav() * self._initial_capital()).rename("portfolio_value")

    def _cash_curve(self) -> pd.Series:
        curve = self.result.meta.get("cash_curve")
        if isinstance(curve, pd.Series):
            return curve.sort_index()
        return pd.Series(0.0, index=self._strategy_nav().index, name="cash")

    def _benchmark_map(self) -> dict[str, pd.Series]:
        if self.benchmark_returns is None:
            return {}
        if isinstance(self.benchmark_returns, pd.Series):
            name = self.benchmark_returns.name or "benchmark"
            return {
                str(name): self.benchmark_returns.reindex(self._strategy_returns().index).fillna(0.0)
            }

        benchmark_df = self.benchmark_returns.reindex(self._strategy_returns().index).fillna(0.0)
        return {
            str(col): benchmark_df[col].astype(float)
            for col in benchmark_df.columns
        }

    def _benchmark_nav_map(self) -> dict[str, pd.Series]:
        return {
            name: (1.0 + returns).cumprod().rename(f"{name}_nav")
            for name, returns in self._benchmark_map().items()
        }

    def _excess_returns_map(self) -> dict[str, pd.Series]:
        strategy_returns = self._strategy_returns()
        return {
            name: (strategy_returns - bench).rename(f"excess_returns_{name}")
            for name, bench in self._benchmark_map().items()
        }

    def _drawdown_duration(self, drawdown: pd.Series) -> int:
        max_duration = 0
        current = 0
        for value in drawdown.fillna(0.0):
            if value < 0:
                current += 1
                max_duration = max(max_duration, current)
            else:
                current = 0
        return int(max_duration)

    def _annual_return(self, returns: pd.Series) -> float:
        if len(returns) == 0:
            return 0.0
        equity = float((1.0 + returns).prod())
        if equity <= 0:
            return -1.0
        return float(equity ** (self.trading_days / len(returns)) - 1.0)

    def _annual_volatility(self, returns: pd.Series) -> float:
        if len(returns) == 0:
            return 0.0
        return float(returns.std(ddof=0) * np.sqrt(self.trading_days))

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        vol = returns.std(ddof=0)
        if len(returns) == 0 or vol == 0 or np.isnan(vol):
            return 0.0
        daily_rf = self.risk_free / self.trading_days
        return float((returns.mean() - daily_rf) / vol * np.sqrt(self.trading_days))

    def _profit_loss_ratio(self, returns: pd.Series) -> float:
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        if losses.empty:
            return float("inf") if not gains.empty else 0.0
        if gains.empty:
            return 0.0
        return float(gains.mean() / abs(losses.mean()))

    def _price_frame(self) -> pd.DataFrame | None:
        if self.market_data is None:
            return None
        prices = (
            self.market_data[self._resolve_price_col()]
            .unstack(Col.SYMBOL)
            .sort_index()
        )

        position_weights = self.result.meta.get("position_weights")
        if isinstance(position_weights, pd.DataFrame):
            prices = prices.reindex(index=position_weights.index, columns=position_weights.columns)
        else:
            positions = self.result.positions.sort_index()
            prices = prices.reindex(index=positions.index, columns=positions.columns)
        return prices.dropna(how="all")

    def _portfolio(self):
        if self._pf is not None:
            return self._pf

        vbt = _ensure_vectorbt()
        prices = self._price_frame()
        if prices is None or prices.empty:
            synthetic = self._strategy_nav() * 100.0
            self._pf = vbt.Portfolio.from_orders(
                close=synthetic,
                size=1.0,
                size_type="targetpercent",
                direction="longonly",
                freq="D",
                init_cash=100.0,
                fees=0.0,
            )
            return self._pf

        position_weights = self.result.meta.get("position_weights")
        if isinstance(position_weights, pd.DataFrame):
            weights = position_weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
        else:
            weights = self.result.positions.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

        commission = _safe_float(self.result.meta.get("commission_rate", 0.0))
        slippage = _safe_float(self.result.meta.get("slippage_rate", 0.0))

        try:
            self._pf = vbt.Portfolio.from_orders(
                close=prices,
                size=weights,
                size_type="targetpercent",
                direction="longonly",
                freq="D",
                init_cash=1.0,
                fees=commission,
                slippage=slippage,
            )
        except Exception as exc:
            logger.warning("构建 vectorbt Portfolio 失败，回退到合成净值组合: {}", exc)
            synthetic = self._strategy_nav() * 100.0
            self._pf = vbt.Portfolio.from_orders(
                close=synthetic,
                size=1.0,
                size_type="targetpercent",
                direction="longonly",
                freq="D",
                init_cash=100.0,
                fees=0.0,
            )
        return self._pf

    def _returns_portfolio(self, name: str, returns: pd.Series | None):
        if returns is None or returns.empty:
            return None
        if name in self._benchmark_pfs:
            return self._benchmark_pfs[name]

        vbt = _ensure_vectorbt()
        synthetic = pd.concat(
            [
                pd.Series([100.0], index=[returns.index[0] - pd.Timedelta(days=1)]),
                (1.0 + returns).cumprod() * 100.0,
            ]
        )
        pf = vbt.Portfolio.from_orders(
            close=synthetic,
            size=1.0,
            size_type="targetpercent",
            direction="longonly",
            freq="D",
            init_cash=100.0,
            fees=0.0,
        )
        self._benchmark_pfs[name] = pf
        return pf

    def nav_figure(self) -> go.Figure:
        strategy_nav = self._strategy_nav()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=strategy_nav.index,
                y=strategy_nav.values,
                mode="lines",
                name=self.strategy_name,
            )
        )
        for name, benchmark_nav in self._benchmark_nav_map().items():
            fig.add_trace(
                go.Scatter(
                    x=benchmark_nav.index,
                    y=benchmark_nav.values,
                    mode="lines",
                    name=name,
                )
            )
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="NAV",
            template="plotly_white",
        )
        return fig

    def drawdown_figure(self) -> go.Figure:
        nav = self._strategy_nav()
        drawdown = nav / nav.cummax() - 1.0
        fig = go.Figure(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill="tozeroy",
                mode="lines",
                name="Drawdown",
            )
        )
        fig.update_layout(
            title="Underwater",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            template="plotly_white",
        )
        return fig

    def monthly_returns_figure(self) -> go.Figure:
        monthly = self._strategy_returns().resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
        if monthly.empty:
            return go.Figure()

        heatmap_df = monthly.to_frame("return")
        heatmap_df["year"] = heatmap_df.index.year
        heatmap_df["month"] = heatmap_df.index.month
        pivot = heatmap_df.pivot(index="year", columns="month", values="return").sort_index()
        pivot = pivot.reindex(columns=range(1, 13))

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[f"{m:02d}" for m in pivot.columns],
                y=pivot.index.astype(str),
                colorscale="RdYlGn",
                zmid=0.0,
                colorbar_title="Return",
            )
        )
        fig.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_white",
        )
        return fig

    def rolling_sharpe_figure(self, window: int = 60) -> go.Figure:
        returns = self._strategy_returns()
        daily_rf = self.risk_free / self.trading_days
        rolling_std = returns.rolling(window).std(ddof=0)
        rolling_sharpe = (
            (returns.rolling(window).mean() - daily_rf)
            / rolling_std.replace(0.0, np.nan)
            * np.sqrt(self.trading_days)
        )
        fig = go.Figure(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode="lines",
                name=f"Rolling Sharpe {window}D",
            )
        )
        fig.add_hline(y=0.0, line_width=1, line_color="gray")
        fig.update_layout(
            title=f"Rolling Sharpe ({window}D)",
            xaxis_title="Date",
            yaxis_title="Sharpe",
            template="plotly_white",
        )
        return fig

    def turnover_figure(self) -> go.Figure:
        trades = self.result.trades
        if trades is None or "turnover" not in trades.columns:
            return go.Figure()
        fig = go.Figure(
            go.Bar(
                x=trades.index,
                y=trades["turnover"].values,
                name="Turnover",
                marker_color="#1f77b4",
            )
        )
        fig.update_layout(
            title="Rebalance Turnover",
            xaxis_title="Date",
            yaxis_title="Turnover",
            template="plotly_white",
        )
        return fig

    def positions_figure(self) -> go.Figure:
        positions = self.result.meta.get("position_weights")
        if not isinstance(positions, pd.DataFrame):
            positions = self.result.positions.fillna(0.0)
        if positions.empty:
            return go.Figure()
        fig = go.Figure(
            data=go.Heatmap(
                z=positions.T.values,
                x=positions.index,
                y=positions.columns.astype(str),
                colorscale="Blues",
                colorbar_title="Weight",
            )
        )
        fig.update_layout(
            title="Position Heatmap",
            xaxis_title="Date",
            yaxis_title="Symbol",
            template="plotly_white",
        )
        return fig

    def portfolio_overview_figure(self):
        portfolio_value = self._portfolio_value_curve()
        cash_value = self._cash_curve()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode="lines",
                name="Portfolio Value",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cash_value.index,
                y=cash_value.values,
                mode="lines",
                name="Cash",
            )
        )
        fig.update_layout(
            template="plotly_white",
            title="Portfolio Value And Cash",
            xaxis_title="Date",
            yaxis_title="Amount",
        )
        return fig

    def rebalance_records(self) -> pd.DataFrame:
        """Return human-readable per-symbol trade records."""
        trade_details = self.result.meta.get("trade_details")
        if isinstance(trade_details, pd.DataFrame) and not trade_details.empty:
            records = trade_details.copy()
            if "cash_after" in records.columns:
                records = records.rename(columns={"cash_after": "balance"})
            ordered_cols = [
                Col.DATE,
                Col.SYMBOL,
                "action",
                "shares",
                "price",
                "notional",
                "balance",
            ]
            records = records[[col for col in ordered_cols if col in records.columns]]
            return records.reset_index(drop=True)

        trades = self.result.trades
        if trades is None or trades.empty:
            return pd.DataFrame()

        records = trades.sort_index().copy()
        activity_mask = pd.Series(False, index=records.index)
        for col in ("trade_notional", "buy_shares", "sell_shares", "turnover"):
            if col in records.columns:
                activity_mask |= records[col].fillna(0.0).abs().gt(0.0)

        records = records.loc[activity_mask]
        records.index.name = Col.DATE
        records = records.reset_index()
        if "cash" in records.columns:
            records = records.rename(columns={"cash": "balance"})
        ordered_cols = [
            Col.DATE,
            "trade_notional",
            "buy_shares",
            "sell_shares",
            "balance",
        ]
        return records[[col for col in ordered_cols if col in records.columns]]

    def _rebalance_records_html(self, records: pd.DataFrame) -> str:
        if records.empty:
            return "<p>No rebalance records.</p>"

        date_group: dict[str, int] = {}
        next_group = 0
        rows: list[str] = []
        for _, row in records.iterrows():
            date_key = str(row.get(Col.DATE, ""))
            if date_key not in date_group:
                date_group[date_key] = next_group
                next_group += 1
            cls = "date-band-odd" if date_group[date_key] % 2 else "date-band-even"
            cells = "".join(
                f"<td>{_format_table_value(row[col])}</td>"
                for col in records.columns
            )
            rows.append(f"<tr class='{cls}'>{cells}</tr>")

        headers = "".join(f"<th>{escape(str(col))}</th>" for col in records.columns)
        return (
            "<table border='1' class='dataframe rebalance-table'>"
            f"<thead><tr>{headers}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )

    def summary_stats(self) -> dict[str, float]:
        if self._stats_cache is not None:
            return self._stats_cache

        returns = self._strategy_returns()
        nav = self._strategy_nav()
        drawdown = nav / nav.cummax() - 1.0
        max_drawdown = float(abs(drawdown.min())) if len(drawdown) else 0.0

        pf = self._portfolio()
        daily_rf = self.risk_free / self.trading_days

        annual_return = _safe_float(
            pf.annualized_return(year_freq=_year_freq(self.trading_days)),
            default=self._annual_return(returns),
        )
        sharpe_ratio = _safe_float(
            pf.sharpe_ratio(risk_free=daily_rf, year_freq=_year_freq(self.trading_days)),
            default=self._sharpe_ratio(returns),
        )
        calmar_ratio = _safe_float(
            pf.calmar_ratio(year_freq=_year_freq(self.trading_days)),
            default=(annual_return / max_drawdown if max_drawdown > 0 else 0.0),
        )

        stats: dict[str, float] = {
            "initial_capital": self._initial_capital(),
            "ending_value": float(self._portfolio_value_curve().iloc[-1]) if len(nav) else self._initial_capital(),
            "total_return": float(nav.iloc[-1] - 1.0) if len(nav) else 0.0,
            "annual_return": annual_return,
            "annual_volatility": self._annual_volatility(returns),
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": _safe_float(abs(pf.max_drawdown()), default=max_drawdown),
            "max_drawdown_duration": float(self._drawdown_duration(drawdown)),
            "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
            "profit_loss_ratio": self._profit_loss_ratio(returns),
            "turnover": float(self.result.meta.get("avg_turnover", 0.0)),
        }

        for name, bench_returns in self._benchmark_map().items():
            excess = (returns - bench_returns).rename(f"excess_{name}")
            excess_equity = (1.0 + excess).cumprod()
            excess_drawdown = excess_equity / excess_equity.cummax() - 1.0
            excess_pf = self._returns_portfolio(f"excess::{name}", excess)
            stats.update(
                {
                    f"excess_annual_return_{name}": _safe_float(
                        excess_pf.annualized_return(year_freq=_year_freq(self.trading_days))
                        if excess_pf is not None else None,
                        default=self._annual_return(excess),
                    ),
                    f"information_ratio_{name}": self._sharpe_ratio(excess),
                    f"excess_max_drawdown_{name}": float(abs(excess_drawdown.min())) if len(excess_drawdown) else 0.0,
                }
            )

        self._stats_cache = stats
        return stats

    def export_report(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for legacy_html in output_dir.glob("*.html"):
            legacy_html.unlink(missing_ok=True)

        figures = [
            ("nav", "Equity Curve", self.nav_figure()),
            ("drawdown", "Underwater", self.drawdown_figure()),
            ("monthly_returns", "Monthly Returns", self.monthly_returns_figure()),
            ("rolling_sharpe", "Rolling Sharpe", self.rolling_sharpe_figure()),
            ("turnover", "Turnover", self.turnover_figure()),
            ("positions", "Position Heatmap", self.positions_figure()),
            ("portfolio_overview", "Portfolio Overview", self.portfolio_overview_figure()),
        ]

        stats_df = pd.Series(self.summary_stats(), name="value").to_frame()
        stats_df.to_csv(output_dir / "summary_stats.csv")
        stats = self.summary_stats()
        summary_cards = [
            ("Total Return", f"{stats.get('total_return', 0.0):.2%}"),
            ("Annual Return", f"{stats.get('annual_return', 0.0):.2%}"),
            ("Win Rate", f"{stats.get('win_rate', 0.0):.2%}"),
            ("Sharpe", f"{stats.get('sharpe_ratio', 0.0):.2f}"),
            ("Max Drawdown", f"{stats.get('max_drawdown', 0.0):.2%}"),
            ("Ending Value", f"{stats.get('ending_value', 0.0):,.2f}"),
        ]
        cards_html = "".join(
            "<div class='metric-card'>"
            f"<div class='metric-label'>{escape(label)}</div>"
            f"<div class='metric-value'>{escape(value)}</div>"
            "</div>"
            for label, value in summary_cards
        )

        pf = self._portfolio()
        vectorbt_stats = pd.Series(
            {
                "annualized_return": _safe_float(
                    pf.annualized_return(year_freq=_year_freq(self.trading_days))
                ),
                "sharpe_ratio": _safe_float(
                    pf.sharpe_ratio(
                        risk_free=self.risk_free / self.trading_days,
                        year_freq=_year_freq(self.trading_days),
                    )
                ),
                "calmar_ratio": _safe_float(
                    pf.calmar_ratio(year_freq=_year_freq(self.trading_days))
                ),
                "max_drawdown": _safe_float(abs(pf.max_drawdown())),
            },
            name="value",
        )
        vectorbt_stats.to_csv(output_dir / "vectorbt_stats.csv", header=True)
        rebalance_records = self.rebalance_records()
        rebalance_records.to_csv(output_dir / "rebalance_records.csv", index=False)
        rebalance_records_html = self._rebalance_records_html(rebalance_records)

        tab_buttons = [
            "<button class='tab-link active' data-tab='summary'>Summary</button>",
            "<button class='tab-link' data-tab='vbt_stats'>vectorbt Stats</button>",
            "<button class='tab-link' data-tab='rebalance_records'>Rebalance Records</button>",
        ]
        tab_panels = [
            (
                "<section id='tab-summary' class='tab-panel active'>"
                "<h2>Summary Stats</h2>"
                f"<div class='metric-grid'>{cards_html}</div>"
                f"{stats_df.to_html()}"
                "</section>"
            ),
            (
                "<section id='tab-vbt_stats' class='tab-panel'>"
                "<h2>vectorbt Stats</h2>"
                f"{vectorbt_stats.to_frame().to_html()}"
                "</section>"
            ),
            (
                "<section id='tab-rebalance_records' class='tab-panel'>"
                "<h2>Rebalance Records</h2>"
                "<p class='muted'>Per-symbol trade records with action, shares, execution price, notional, and cash balance after the trade.</p>"
                "<div class='table-wrap'>"
                f"{rebalance_records_html}"
                "</div>"
                "</section>"
            ),
        ]

        chart_specs: dict[str, dict] = {}
        for tab_id, title, fig in figures:
            tab_buttons.append(
                f"<button class='tab-link' data-tab='{tab_id}'>{title}</button>"
            )
            chart_specs[tab_id] = json.loads(pio.to_json(fig, pretty=False))
            tab_panels.append(
                f"<section id='tab-{tab_id}' class='tab-panel'>"
                f"<h2>{title}</h2><div id='plot-{tab_id}' style='height:650px;'></div></section>"
            )

        chart_specs_json = json.dumps(chart_specs)

        report_path = output_dir / "report.html"
        report_path.write_text(
            (
                "<html><head><meta charset='utf-8'><title>Strategy Report</title>"
                "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
                "<style>"
                "body{font-family:Arial,sans-serif;margin:24px;}"
                ".metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0 20px 0;}"
                ".metric-card{border:1px solid #d0d7de;border-radius:10px;padding:14px;background:#f6f8fa;}"
                ".metric-label{color:#57606a;font-size:13px;margin-bottom:6px;}"
                ".metric-value{font-size:22px;font-weight:700;color:#24292f;}"
                "table{border-collapse:collapse;}td,th{border:1px solid #ddd;padding:8px;}"
                ".table-wrap{max-height:680px;overflow:auto;border:1px solid #ddd;}"
                ".table-wrap table{width:100%;}"
                ".table-wrap th{position:sticky;top:0;background:#f6f8fa;z-index:1;}"
                ".rebalance-table tr.date-band-even td{background:#ffffff;}"
                ".rebalance-table tr.date-band-odd td{background:#f3f6fa;}"
                ".rebalance-table tr:hover td{background:#fff8c5;}"
                ".muted{color:#57606a;}"
                ".tabs{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 20px 0;}"
                ".tab-link{border:1px solid #d0d7de;background:#f6f8fa;padding:8px 12px;cursor:pointer;border-radius:8px;}"
                ".tab-link.active{background:#0969da;color:#fff;border-color:#0969da;}"
                ".tab-panel{display:none;}"
                ".tab-panel.active{display:block;}"
                "</style>"
                "</head><body>"
                f"<h1>{self.strategy_name} Report</h1>"
                "<div class='tabs'>"
                + "".join(tab_buttons)
                + "</div>"
                + "".join(tab_panels)
                + "<script>"
                "const links=document.querySelectorAll('.tab-link');"
                "const panels=document.querySelectorAll('.tab-panel');"
                f"const chartSpecs={chart_specs_json};"
                "const renderedCharts={};"
                "function renderChart(tabId){"
                "if(!chartSpecs[tabId]||renderedCharts[tabId]) return;"
                "const spec=chartSpecs[tabId];"
                "Plotly.newPlot('plot-'+tabId,spec.data||[],spec.layout||{},"
                "{responsive:true,displaylogo:false,scrollZoom:true});"
                "renderedCharts[tabId]=true;"
                "}"
                "links.forEach(btn=>btn.addEventListener('click',()=>{"
                "links.forEach(x=>x.classList.remove('active'));"
                "panels.forEach(x=>x.classList.remove('active'));"
                "btn.classList.add('active');"
                "const tabId=btn.dataset.tab;"
                "const p=document.getElementById('tab-'+tabId);"
                "if(p){p.classList.add('active');}"
                "renderChart(tabId);"
                "}));"
                "</script>"
                + "</body></html>"
            ),
            encoding="utf-8",
        )
        return report_path

    def plot_tearsheet(
        self,
        output_dir: str | Path | None = None,
        show: bool = False,
    ) -> None:
        if output_dir is not None:
            report_path = self.export_report(output_dir)
            logger.info("策略报告已导出: {}", report_path)
        if show:
            self.nav_figure().show()
            self.drawdown_figure().show()
            self.monthly_returns_figure().show()
            self.rolling_sharpe_figure().show()
