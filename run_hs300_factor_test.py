"""
沪深300因子测试脚本
===================
基于 AKShare 数据，对沪深300成分股测试以下三个因子：
  - momentum_5   （5日动量）
  - momentum_20  （20日动量）
  - volatility_20（20日波动率）

用法：
  python run_hs300_factor_test.py

关键参数（直接修改下方 CONFIG 区域）：
  START_DATE   : 数据起始日期
  END_DATE     : 数据截止日期
  MAX_STOCKS   : 最多使用多少只股票（None = 全部300只，首次运行建议先用 50）
  N_GROUPS     : 分层回测分组数
  SAVE_PLOTS   : 是否保存因子报告图到 outputs/ 目录
"""
from __future__ import annotations

import sys
import matplotlib
matplotlib.use("Agg")  # 必须在导入 pyplot 前设置，避免无头环境弹窗

from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import akshare as ak
from loguru import logger

from config import get_config
from data.akshare_loader import AKShareLoader
from evaluation.plot import plot_factor_report
from pipeline.runner import FactorPipeline

# 从配置文件读取运行参数
_cfg        = get_config()
_data_cfg   = _cfg.get("data", {})
_bt_cfg     = _cfg.get("backtest", {})
_out_cfg    = _cfg.get("output", {})

_eval_cfg   = _cfg.get("evaluation", {})

START_DATE  = _bt_cfg.get("start_date", "20240101")
END_DATE    = _bt_cfg.get("end_date",   "20260101")
MAX_STOCKS  = _data_cfg.get("max_stocks", None)   # None = 不限制
SAVE_PLOTS  = _out_cfg.get("save_plots", True)
OUTPUT_DIR  = _out_cfg.get("dir", "outputs")
PLOT_PERIOD = _eval_cfg.get("plot_period", None)  # None = 自动取 forward_periods[1]


# ── 1. 获取沪深300成分股列表 ─────────────────────────────────────────────

def get_hs300_symbols(max_stocks: int | None = None) -> list[str]:
    logger.info("正在获取沪深300成分股列表...")
    df = ak.index_stock_cons_csindex(symbol="000300")
    symbols = df["成分券代码"].tolist()
    if max_stocks is not None:
        symbols = symbols[:max_stocks]
    logger.info("共获取 {} 只成分股", len(symbols))
    return symbols


# ── 2. 主流程 ────────────────────────────────────────────────────────────

def main():
    factors = ["momentum_5", "momentum_20", "volatility_20"]
    output_dir = Path(OUTPUT_DIR) if SAVE_PLOTS else None

    # 获取成分股列表
    symbols = get_hs300_symbols(max_stocks=MAX_STOCKS)

    # 构建流水线：AKShareLoader 负责下载 + 缓存，FactorPipeline 负责评估
    pipeline = (
        FactorPipeline()
        .set_data_loader(AKShareLoader())
        .add_factors(factors)
    )

    # run() 内部调用 loader.load_market_data(symbols, start, end)
    # 返回 dict[factor_name, FactorReport]
    reports = pipeline.run(
        symbols=symbols,
        start=START_DATE,
        end=END_DATE,
        show_plot=False,  # 手动保存，不弹窗
    )

    # ── 3. 打印汇总 + 保存图表 ───────────────────────────────────────────
    for name, report in reports.items():
        report.print()

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            cfg_periods = report.forward_periods
            if PLOT_PERIOD is not None and PLOT_PERIOD in cfg_periods:
                plot_period = PLOT_PERIOD
            else:
                plot_period = cfg_periods[1] if len(cfg_periods) > 1 else cfg_periods[0]

            ic_s  = report.ic_series(plot_period)
            lr    = report.layered(plot_period)
            decay = report.ic_decay()

            fig = plot_factor_report(ic_s, lr, decay, factor_name=name, period=plot_period)
            save_path = output_dir / f"{name}_report.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("图表已保存: {}", save_path)

    # ── 4. 汇总 CSV ──────────────────────────────────────────────────────
    if output_dir is not None:
        import pandas as pd
        rows = []
        for name, report in reports.items():
            df = report.summary()
            df.insert(0, "factor", name)
            rows.append(df)
        summary = pd.concat(rows)
        csv_path = output_dir / "factor_summary.csv"
        summary.to_csv(csv_path)
        logger.info("汇总表已保存: {}", csv_path)
        print("\n" + "=" * 55)
        print("  综合汇总（所有因子 × 所有周期）")
        print("=" * 55)
        print(summary.to_string())


if __name__ == "__main__":
    main()
