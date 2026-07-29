"""因子评估结果可视化。"""

import os
from pathlib import Path

matplotlib_config_dir = Path("/tmp") / f"matplotlib-{os.getuid()}"
matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factors_eval import FactorEvalResult


YEARLY_SUMMARY_METRICS = (
    "IC_mean",
    "ICIR",
    "timing_IC_mean",
    "top_20%_cumulative_return",
    "bottom_20%_cumulative_return",
    "top_20%_sharpe_ratio",
    "bottom_20%_sharpe_ratio",
    "top_20%_bottom_20%_win_rate",
)


def save_yearly_summary_trends(
    yearly_summary: pd.DataFrame,
    output_path: str | Path,
    factor_name: str,
    dpi: int = 150,
) -> Path:
    """绘制年度 ``factor_summary.csv`` 指定指标的逐年变化。

    输入表每行对应一个自然年和一个前向周期。绘图直接使用原始数值，
    不取绝对值，以保留指标的方向信息。
    """
    required_columns = {"year", *YEARLY_SUMMARY_METRICS}
    missing_columns = required_columns.difference(yearly_summary.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"yearly_summary 缺少绘图字段: {missing}")

    values = yearly_summary.copy().sort_values("year")
    years = values["year"].astype(str).tolist()
    positions = np.arange(len(values))
    figure, axes = plt.subplots(4, 2, figsize=(16, 16), sharex=True)

    for axis, metric in zip(axes.flat, YEARLY_SUMMARY_METRICS):
        metric_values = pd.to_numeric(values[metric], errors="coerce")
        axis.plot(
            positions,
            metric_values,
            marker="o",
            linewidth=1.5,
            color="steelblue",
        )
        axis.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        axis.set_title(metric)
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)

    for axis in axes[-1, :]:
        axis.set_xticks(positions)
        axis.set_xticklabels(years, rotation=45, ha="right")
        axis.set_xlabel("Year")

    figure.suptitle(
        f"Yearly Factor Summary Trends: {factor_name}",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


class FactorPlotter:
    """绘制 ``eval()`` 返回的完整因子评估结果。"""

    def __init__(
        self,
        result: FactorEvalResult,
        factor_name: str = "Factor",
        benchmark_cumulative: pd.DataFrame | None = None,
    ) -> None:
        self.result = result
        self.factor_name = factor_name
        self.benchmark_cumulative = benchmark_cumulative

    def plot_ic_series(self, ax: plt.Axes, rolling_window: int = 20) -> None:
        """绘制 IC 时间序列及滚动均值。"""
        ic = self.result.sampled_ic_series.dropna()
        ax.set_title("IC Time Series")
        if ic.empty:
            return
        ax.fill_between(
            ic.index, 0, ic.values,
            where=ic.values >= 0,
            color="#2c7bb6",
            alpha=0.35,
            label="IC > 0",
        )
        ax.fill_between(
            ic.index, 0, ic.values,
            where=ic.values < 0,
            color="#d7191c",
            alpha=0.35,
            label="IC < 0",
        )
        rolling_mean = ic.rolling(rolling_window).mean()
        ax.plot(rolling_mean.index, rolling_mean, color="black", label=f"MA({rolling_window})")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("IC")
        ax.legend(fontsize=8)

    def plot_layered_returns(self, ax: plt.Axes) -> None:
        """绘制各因子分组的累计收益。"""
        cumulative = (1 + self.result.layered.group_returns).cumprod() - 1
        ax.set_title("Layered Cumulative Returns")
        if cumulative.empty:
            return
        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(cumulative.columns)))
        for color, group in zip(colors, cumulative.columns):
            ax.plot(
                cumulative.index,
                cumulative[group],
                color=color,
                label=f"Group {group}",
            )
        if self.benchmark_cumulative is not None:
            benchmark_colors = {
                "HS300": "black",
                "ZZ500": "dimgray",
                "ZZ1000": "steelblue",
                "SPY": "black",
                "QQQ": "dimgray",
            }
            for benchmark in self.benchmark_cumulative.columns:
                ax.plot(
                    self.benchmark_cumulative.index,
                    self.benchmark_cumulative[benchmark],
                    color=benchmark_colors.get(benchmark, "black"),
                    linestyle="--",
                    linewidth=1.5,
                    label=benchmark,
                )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Cumulative Return")
        ax.legend(fontsize=8)

    def plot_turnover(self, ax: plt.Axes) -> None:
        """绘制最高因子分组组合的换手率。"""
        turnover = self.result.turnover.dropna()
        ax.set_title("Portfolio Turnover")
        if turnover.empty:
            return
        ax.plot(turnover.index, turnover.values, color="darkorange", linewidth=1.0)
        ax.axhline(turnover.mean(), color="black", linestyle="--", label=f"Mean={turnover.mean():.4f}")
        ax.set_ylabel("Turnover")
        ax.legend(fontsize=8)

    def plot_summary(self, ax: plt.Axes) -> None:
        """以表格形式展示核心汇总指标。"""
        summary = self.result.summary.reset_index()
        formatted = summary.copy()
        for column in formatted.columns:
            if column != "period":
                formatted[column] = formatted[column].map(
                    lambda value: "NaN" if not np.isfinite(value) else f"{value:.4f}"
                )
        ax.axis("off")
        ax.set_title("Summary")
        table = ax.table(
            cellText=formatted.T.values,
            rowLabels=formatted.columns,
            colLabels=["Value"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)

    def plot(self, figsize: tuple[float, float] = (16, 10)) -> plt.Figure:
        """生成包含全部评估图表的综合报告图。"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        period = self.result.summary.index[0]
        fig.suptitle(
            f"Factor Report: {self.factor_name} | Forward Period: {period}d",
            fontsize=15,
            fontweight="bold",
        )
        self.plot_ic_series(axes[0, 0])
        self.plot_layered_returns(axes[0, 1])
        self.plot_turnover(axes[1, 0])
        self.plot_summary(axes[1, 1])
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    def save(self, output_path: str | Path, dpi: int = 150) -> Path:
        """生成综合报告图并保存到指定路径。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure = self.plot()
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return output_path
