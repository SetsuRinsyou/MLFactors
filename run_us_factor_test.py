"""
美股因子测试脚本（本地 SQLite 数据）
====================================
基于本地 ``cta_orange.db``，对美股测试以下三个选股因子：
  - momentum_5   （5日动量）
  - momentum_20  （20日动量）
  - volatility_20（20日波动率）

用法：
  ./.venv/bin/python run_us_factor_test.py

常用参数：
  --db-path      SQLite 数据库路径
  --start / --end 回测区间
  --max-stocks   最多使用多少只股票，0 表示不限制
  --output-dir   报告输出目录
  --no-plots     只输出汇总 CSV，不保存图表
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from config import get_config
from data.local_loader import USStockLocalLoader
from evaluation.plot import plot_factor_report
from pipeline.selection_runner import SelectionPipeline


_cfg = get_config()
_eval_cfg = _cfg.get("evaluation", {})

DEFAULT_FACTORS = ["momentum_5", "momentum_20", "volatility_20"]
DEFAULT_DB_PATH = Path("/home/setsu/workspace/data/cta_orange.db")
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2026-01-01"
DEFAULT_OUTPUT_DIR = Path("outputs/us_factor")
DEFAULT_PLOT_PERIOD = _eval_cfg.get("plot_period", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行美股选股因子级别测试")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--market", default="US")
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=DEFAULT_END_DATE)
    parser.add_argument("--max-stocks", type=int, default=300, help="0 表示不限制")
    parser.add_argument("--min-observations", type=int, default=80)
    parser.add_argument("--factors", nargs="+", default=DEFAULT_FACTORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-period", type=int, default=DEFAULT_PLOT_PERIOD)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def get_symbols(
    db_path: Path,
    market: str,
    start: str,
    end: str,
    max_stocks: int | None = None,
    min_observations: int = 80,
) -> list[str]:
    """从 SQLite bars 表读取区间内有足够日线数据的美股代码。"""
    if not db_path.exists():
        logger.error("SQLite 数据库不存在: {}", db_path)
        return []

    conditions = ["market = ?", "frequency = '1d'"]
    params: list[object] = [market]
    if start:
        conditions.append("date(dt) >= ?")
        params.append(str(pd.Timestamp(start).date()))
    if end:
        conditions.append("date(dt) <= ?")
        params.append(str(pd.Timestamp(end).date()))

    query = f"""
        SELECT symbol, COUNT(*) AS n_obs
        FROM bars
        WHERE {" AND ".join(conditions)}
        GROUP BY symbol
        HAVING n_obs >= ?
        ORDER BY symbol
    """
    params.append(int(min_observations))
    if max_stocks is not None and max_stocks > 0:
        query += " LIMIT ?"
        params.append(int(max_stocks))

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(query, params).fetchall()

    symbols = [row[0] for row in rows]
    logger.info(
        "共加载 {} 只美股（market={}, 区间 {} ~ {}, min_observations={}）",
        len(symbols),
        market,
        start,
        end,
        min_observations,
    )
    return symbols


def save_reports(
    reports: dict,
    output_dir: Path,
    save_plots: bool,
    plot_period: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, report in reports.items():
        summary_df = report.summary()
        summary_df.insert(0, "factor", name)
        rows.append(summary_df)

        if save_plots:
            cfg_periods = report.forward_periods
            if plot_period is not None and plot_period in cfg_periods:
                selected_period = plot_period
            else:
                selected_period = cfg_periods[1] if len(cfg_periods) > 1 else cfg_periods[0]

            ic_s = report.ic_series(selected_period)
            layered = report.layered(selected_period)
            decay = report.ic_decay()

            fig = plot_factor_report(ic_s, layered, decay, factor_name=name, period=selected_period)
            save_path = output_dir / f"{name}_report.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("图表已保存: {}", save_path)

    summary = pd.concat(rows)
    csv_path = output_dir / "factor_summary.csv"
    summary.to_csv(csv_path)
    logger.info("汇总表已保存: {}", csv_path)

    print("\n" + "=" * 55)
    print("  美股因子综合汇总（所有因子 x 所有周期）")
    print("=" * 55)
    print(summary.to_string())


def main() -> int:
    args = parse_args()
    max_stocks = None if args.max_stocks == 0 else args.max_stocks

    symbols = get_symbols(
        db_path=args.db_path,
        market=args.market,
        start=args.start,
        end=args.end,
        max_stocks=max_stocks,
        min_observations=args.min_observations,
    )
    if not symbols:
        logger.error("股票列表为空，退出")
        return 1

    selected_symbols_path = args.output_dir / "selected_symbols.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(symbols, name="symbol").to_csv(selected_symbols_path, index=False)
    logger.info("股票列表已保存: {}", selected_symbols_path)

    pipeline = (
        SelectionPipeline()
        .set_data_loader(USStockLocalLoader(db_path=args.db_path, market=args.market))
        .add_factors(args.factors)
    )

    reports = pipeline.run(
        symbols=symbols,
        start=args.start,
        end=args.end,
        show_plot=False,
    )

    save_reports(
        reports=reports,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
        plot_period=args.plot_period,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
