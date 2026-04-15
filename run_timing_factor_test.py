"""
择时因子测试脚本
================
基于本地合成数据（或 AKShare 真实数据），对目标股票测试以下择时因子：
  - ma_cross_timing  （双均线交叉）
  - rsi_timing       （RSI 超买超卖反转）

用法：
  python run_timing_factor_test.py

关键参数（直接修改下方 CONFIG 区域）：
  USE_AKSHARE  : True = 使用 AKShare 真实数据，False = 使用本地合成数据
  SYMBOLS      : 目标股票代码列表
  BENCHMARK    : 基准指数代码（None = 无基准）
  START_DATE   : 数据起始日期
  END_DATE     : 数据截止日期
  SAVE_PLOTS   : True = 保存图表到 outputs/timing/；False = 直接弹窗显示
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")   # 无头环境下不弹窗；若需要弹窗请改为 "TkAgg" 或注释掉

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger

from data.schema import Col

# ── CONFIG ───────────────────────────────────────────────────────────────────

USE_AKSHARE = False          # False = 使用合成数据快速跑通脚本

SYMBOLS     = ["000001"]     # 目标股票代码
BENCHMARK   = None           # 基准代码，例如 "000300"；None = 无基准

START_DATE  = "20220101"
END_DATE    = "20240101"

SAVE_PLOTS  = True           # True = 保存到 outputs/timing/；False = plt.show()
OUTPUT_DIR  = Path("outputs/timing")

# ── 1. 数据加载 ───────────────────────────────────────────────────────────────

def load_data(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """加载行情数据（合成 or AKShare）。返回 MultiIndex(date, symbol) DataFrame。"""
    if USE_AKSHARE:
        from data.akshare_loader import AKShareLoader
        loader = AKShareLoader()
        all_syms = list(symbols)
        if BENCHMARK and BENCHMARK not in all_syms:
            all_syms.append(BENCHMARK)
        return loader.load_market_data(all_syms, start, end)
    else:
        return _make_synthetic(symbols, start, end)


def _make_synthetic(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """生成合成行情数据（GBM 随机游走），格式与 DataLoader 一致。"""
    dates = pd.bdate_range(start, end)
    frames = []
    rng = np.random.default_rng(42)
    for sym in symbols:
        n = len(dates)
        log_ret = rng.normal(0.0003, 0.012, n)
        close = 100.0 * np.exp(np.cumsum(log_ret))
        open_ = close * (1 + rng.normal(0, 0.003, n))
        high  = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.003, n)))
        low   = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.003, n)))
        df = pd.DataFrame(
            {
                Col.DATE:   dates,
                Col.SYMBOL: sym,
                Col.OPEN:   open_,
                Col.HIGH:   high,
                Col.LOW:    low,
                Col.CLOSE:  close,
                Col.VOLUME: rng.integers(1_000_000, 10_000_000, n).astype(float),
            }
        )
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    result[Col.DATE] = pd.to_datetime(result[Col.DATE])
    result = result.set_index([Col.DATE, Col.SYMBOL]).sort_index()
    return result


# ── 2. 因子计算 ───────────────────────────────────────────────────────────────

def compute_factors(market: pd.DataFrame, symbols: list[str]) -> dict[str, dict[str, pd.Series]]:
    """返回 {factor_name: {symbol: signal_series}}。"""
    from factors.library.timing import ma_cross, rsi  # 确保注册
    from factors.library.timing.ma_cross import MACrossTiming
    from factors.library.timing.rsi import RSITiming

    factor_instances = [
        MACrossTiming(fast=5, slow=20),
        RSITiming(period=14, overbought=70, oversold=30),
    ]

    results: dict[str, dict[str, pd.Series]] = {}
    for factor in factor_instances:
        sym_signals: dict[str, pd.Series] = {}
        for sym in symbols:
            try:
                sig = factor.compute_timing(market, sym)
                sym_signals[sym] = sig
            except Exception as exc:
                logger.warning("因子 '{}' 在 '{}' 失败: {}", factor.name, sym, exc)
        results[factor.name] = sym_signals

    return results


# ── 3. 单因子单股票报告 + 绘图 ────────────────────────────────────────────────

def run_report_and_plot(
    factor_name: str,
    symbol: str,
    signal: pd.Series,
    market: pd.DataFrame,
    output_dir: Path | None,
) -> None:
    """计算 TimingReport，打印指标，并输出 vectorbt 图表。"""
    from evaluation.timimg.report import TimingReport

    bm_data   = market if BENCHMARK else None
    report = TimingReport(
        factor_values=signal,
        market_data=market,
        symbol=symbol,
        benchmark_data=bm_data,
        benchmark_symbol=BENCHMARK,
        trading_days=252,
        risk_free=0.0,
    )
    report.print()

    # —— vectorbt 图表 ——————————————————————————————————————————————————
    pf = report._portfolio()   # 复用已缓存的 vbt.Portfolio

    tag = f"{factor_name}__{symbol}"

    # 3-a  净值曲线（策略 vs 基准买持）
    _plot_nav(report, tag, output_dir)

    # 3-b  vbt 内置组合总览图
    _plot_vbt_portfolio(pf, tag, output_dir)

    # 3-c  成交记录（buy/sell 标注在价格上）
    _plot_trades(pf, tag, output_dir)

    # 3-d  水下回撤图（Underwater plot）
    _plot_underwater(pf, tag, output_dir)


def _save_or_show(fig, path: Path | None) -> None:
    """统一处理保存/显示逻辑，强制全屏自适应。"""
    
    # 1. 核心修复：解除图形自带的固定像素宽高，开启自动缩放
    fig.update_layout(
        autosize=True,
        width=None,    # 清除写死的宽度
        height=None,   # 清除写死的高度
        margin=dict(l=20, r=20, t=40, b=20)  # 缩小四周多余的白边，留出顶部标题空间
    )

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 2. 关键输出配置：在生成 HTML 时，强制把容器的宽高设为 100% 视口大小
        fig.write_html(
            str(path.with_suffix(".html")),
            default_width="100%",      # 铺满浏览器宽度
            default_height="100vh",    # 铺满浏览器高度（vh = viewport height）
            include_plotlyjs="cdn"     # 推荐：使用 CDN 加载 js 库，能让 html 文件从几MB缩小到几百KB
        )
        logger.info("图表已保存: {}", path.with_suffix(".html"))
    else:
        fig.show()


def _save_or_show_mpl(fig, path: Path | None) -> None:
    """统一处理 matplotlib 图的保存/显示逻辑。"""
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path.with_suffix(".png")), dpi=150, bbox_inches="tight")
        logger.info("图表已保存: {}", path.with_suffix(".png"))
        plt.close(fig)
    else:
        plt.show()


def _plot_nav(report, tag: str, output_dir: Path | None) -> None:
    """用 matplotlib 画策略净值 vs 基准净值（折线）。"""
    fig, ax = plt.subplots(figsize=(12, 5))

    nav = report.nav()
    ax.plot(nav.index, nav.values, label="Strategy NAV", linewidth=1.5)

    bm_nav = report.benchmark_nav()
    if bm_nav is not None:
        ax.plot(bm_nav.index, bm_nav.values, label="Benchmark NAV", linewidth=1.2, alpha=0.8)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"{tag} — NAV Curve")
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    _save_or_show_mpl(fig, output_dir / f"{tag}__nav" if output_dir else None)


def _plot_vbt_portfolio(pf: "vbt.Portfolio", tag: str, output_dir: Path | None) -> None:
    """vectorbt 内置组合总览（cum_returns）。"""
    fig = pf.plot()
    fig.update_layout(title=f"{tag} — vectorbt Portfolio 总览")
    _save_or_show(fig, output_dir / f"{tag}__vbt_portfolio" if output_dir else None)


def _plot_trades(pf: "vbt.Portfolio", tag: str, output_dir: Path | None) -> None:
    """vectorbt 成交记录图（价格 + 买卖标注）。"""
    try:
        fig = pf.trades.plot()
        fig.update_layout(title=f"{tag} — 成交记录")
        _save_or_show(fig, output_dir / f"{tag}__trades" if output_dir else None)
    except Exception as exc:
        logger.warning("成交记录图绘制失败（可能无成交）: {}", exc)


def _plot_underwater(pf: "vbt.Portfolio", tag: str, output_dir: Path | None) -> None:
    """vectorbt 水下回撤图。"""
    try:
        drawdowns = pf.drawdowns
        fig = drawdowns.plot()
        fig.update_layout(title=f"{tag} — 水下回撤（Drawdown）")
        _save_or_show(fig, output_dir / f"{tag}__drawdown" if output_dir else None)
    except Exception as exc:
        logger.warning("回撤图绘制失败: {}", exc)


# ── 4. 主函数 ─────────────────────────────────────────────────────────────────

def main() -> None:
    output_dir = OUTPUT_DIR if SAVE_PLOTS else None

    # 1. 加载数据
    logger.info("加载行情数据 ({})...", "AKShare" if USE_AKSHARE else "合成数据")
    market = load_data(SYMBOLS, START_DATE, END_DATE)
    logger.info("行情数据: {} 行，时间范围 {} ~ {}",
                len(market),
                market.index.get_level_values(Col.DATE).min().date(),
                market.index.get_level_values(Col.DATE).max().date())

    # 2. 计算因子信号
    logger.info("计算因子信号...")
    factor_signals = compute_factors(market, SYMBOLS)

    # 3. 逐因子 × 逐股票生成报告 + 图表
    for factor_name, sym_signals in factor_signals.items():
        for symbol, signal in sym_signals.items():
            logger.info("─" * 60)
            logger.info("因子: {}  股票: {}", factor_name, symbol)
            run_report_and_plot(factor_name, symbol, signal, market, output_dir)

    logger.info("=" * 60)
    logger.info("全部完成。图表输出目录: {}", output_dir or "（屏幕显示）")


if __name__ == "__main__":
    main()
