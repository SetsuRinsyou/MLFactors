"""本地文件数据加载器 — 支持 CSV / Parquet / SQLite / A 股标准数据目录。"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from data.base import DataLoader
from data.schema import Col, FundamentalCol

# ====================================================================
#  AStockLocalLoader — 从标准本地数据目录加载 A 股数据
# ====================================================================

class AStockLocalLoader(DataLoader):
    """从本地标准数据目录加载 A 股行情和估值数据。

    目录结构::

        <data_root>/
        ├── meta_data.duckdb             # 元数据（股票列表、交易日历）
        │   ├── securities               # 股票列表（含上市/退市日期）
        │   └── trade_calendar           # 交易日历
        ├── fundamentals.duckdb          # 基本面数据库
        │   ├── daily_valuation          # 每日估值（PE/PB/PS 等）
        │   └── financial_reports        # 财报数据（接口预留）
        └── market_data/
            └── market=A_stock/
                └── year=YYYY/
                    └── data.parquet     # 日线行情（OHLCV）

    股票过滤
    --------
    若 ``meta_data.duckdb`` 存在，加载时自动过滤无效标的：

    - 上市日期晚于 ``end`` 的股票（请求区间内尚未上市）
    - 退市日期早于 ``start`` 的股票（请求区间前已退市）

    若 ``meta_data.duckdb`` 不存在，跳过过滤，返回原始数据。

    Parameters
    ----------
    data_root : 本地数据根目录。
    """

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self._market_dir = self.data_root / "market_data"
        self._fund_db = self.data_root / "fundamentals.duckdb"
        self._meta_db = self.data_root / "meta_data.duckdb"

    # ------------------------------------------------------------------ #
    #  内部：有效股票过滤
    # ------------------------------------------------------------------ #

    def _load_valid_symbols(
        self,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> set[str] | None:
        """从 meta_data.duckdb 读取在给定日期区间内活跃的股票代码集合。

        过滤规则：
        - ``list_date <= end``：截止日前已上市
        - ``delist_date IS NULL OR delist_date >= start``：未在开始日前退市

        Returns
        -------
        活跃 symbol 集合；若 meta_data.duckdb 不存在则返回 ``None``（跳过过滤）。
        """
        if not self._meta_db.exists():
            logger.debug("meta_data.duckdb 不存在，跳过股票有效性过滤")
            return None

        conditions = ["market = 'A_stock'", "asset_type = 'stock'"]
        params: list = []
        if end is not None:
            conditions.append("(list_date IS NULL OR list_date <= ?)")
            params.append(str(pd.Timestamp(end).date()))
        if start is not None:
            conditions.append("(delist_date IS NULL OR delist_date >= ?)")
            params.append(str(pd.Timestamp(start).date()))

        query = f"SELECT symbol FROM securities WHERE {' AND '.join(conditions)}"
        try:
            with duckdb.connect(str(self._meta_db), read_only=True) as conn:
                rows = conn.execute(query, params).fetchall()
            valid = {r[0] for r in rows}
            logger.debug("meta_data: 区间内有效股票 {} 只", len(valid))
            return valid
        except Exception as exc:
            logger.warning("读取 meta_data.duckdb 失败，跳过过滤: {}", exc)
            return None

    # ------------------------------------------------------------------ #
    #  行情：Hive 分区 Parquet
    # ------------------------------------------------------------------ #

    def load_market_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """从 Hive 分区 Parquet 加载日线行情（OHLCV）。

        仅读取与日期区间重叠的年份分区，减少不必要的 I/O。

        Returns
        -------
        DataFrame，以 (date, symbol) 为 MultiIndex，包含
        open / high / low / close / volume / amount / adjust_flag 列。
        """
        partition_root = self._market_dir / "market=A_stock"
        if not partition_root.exists():
            logger.warning("market_data 目录不存在: {}", partition_root)
            return pd.DataFrame()

        start_year = pd.Timestamp(start).year if start is not None else None
        end_year   = pd.Timestamp(end).year   if end   is not None else None

        parquet_files = sorted(partition_root.glob("year=*/data.parquet"))
        if not parquet_files:
            logger.warning("未找到任何 Parquet 分区文件")
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for pq_file in parquet_files:
            try:
                year = int(pq_file.parent.name.split("=")[1])
            except (IndexError, ValueError):
                continue
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue
            try:
                frames.append(pq.read_table(pq_file).to_pandas())
            except Exception as exc:
                logger.warning("读取 {} 失败，跳过: {}", pq_file, exc)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        # trade_date → Col.DATE ("date")
        df = df.rename(columns={"trade_date": Col.DATE})
        df[Col.DATE]   = pd.to_datetime(df[Col.DATE])
        df[Col.SYMBOL] = df[Col.SYMBOL].astype(str)

        # 与 meta_data 中有效股票取交集（过滤已退市或不存在的标的）
        valid = self._load_valid_symbols(start, end)
        if valid is not None:
            effective_symbols = list(valid if symbols is None else set(symbols) & valid)
        else:
            effective_symbols = symbols

        df = self._filter(df, effective_symbols, start, end)
        df = self._set_index(df)

        missing = set(Col.market_required()) - set(df.columns) - set(df.index.names)
        if missing:
            logger.warning("行情数据缺少列: {}", missing)

        return df

    # ------------------------------------------------------------------ #
    #  估值：fundamentals.duckdb → daily_valuation
    # ------------------------------------------------------------------ #

    def load_fundamental_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """从 ``fundamentals.duckdb`` 加载每日估值数据（PE/PB/PS）。

        财报数据请使用 :meth:`load_financial_reports`。

        Returns
        -------
        DataFrame，以 (date, symbol) 为 MultiIndex，包含
        pe / pe_ttm / pb / ps / ps_ttm / total_mv / circ_mv 列。
        """
        if not self._fund_db.exists():
            logger.warning("fundamentals.duckdb 不存在: {}", self._fund_db)
            return pd.DataFrame()

        conditions: list[str] = []
        params: list = []

        if symbols:
            placeholders = ",".join("?" * len(symbols))
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)
        if start is not None:
            conditions.append("trade_date >= ?")
            params.append(str(pd.Timestamp(start).date()))
        if end is not None:
            conditions.append("trade_date <= ?")
            params.append(str(pd.Timestamp(end).date()))

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = (
            "SELECT symbol, trade_date, pe, pe_ttm, pb, ps, ps_ttm, total_mv, circ_mv "
            f"FROM daily_valuation{where} ORDER BY trade_date, symbol"
        )

        # 与 meta_data 有效股票取交集，追加到 SQL 过滤条件中
        valid = self._load_valid_symbols(start, end)
        if valid is not None:
            effective_symbols = list(valid if symbols is None else set(symbols) & valid)
            if effective_symbols != symbols:  # 有过滤动作时重建 WHERE
                conditions = []
                params = []
                placeholders = ",".join("?" * len(effective_symbols))
                conditions.append(f"symbol IN ({placeholders})")
                params.extend(effective_symbols)
                if start is not None:
                    conditions.append("trade_date >= ?")
                    params.append(str(pd.Timestamp(start).date()))
                if end is not None:
                    conditions.append("trade_date <= ?")
                    params.append(str(pd.Timestamp(end).date()))
                where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
                query = (
                    "SELECT symbol, trade_date, pe, pe_ttm, pb, ps, ps_ttm, total_mv, circ_mv "
                    f"FROM daily_valuation{where} ORDER BY trade_date, symbol"
                )

        try:
            with duckdb.connect(str(self._fund_db), read_only=True) as conn:
                df = conn.execute(query, params).df()
        except Exception as exc:
            logger.error("加载 daily_valuation 失败: {}", exc)
            return pd.DataFrame()

        df = df.rename(columns={
            "trade_date": FundamentalCol.DATE,
            "pe":         FundamentalCol.PE,
            "pe_ttm":     FundamentalCol.PE_TTM,
            "pb":         FundamentalCol.PB,
            "ps":         FundamentalCol.PS,
            "ps_ttm":     FundamentalCol.PS_TTM,
        })
        df[FundamentalCol.DATE]   = pd.to_datetime(df[FundamentalCol.DATE])
        df[FundamentalCol.SYMBOL] = df[FundamentalCol.SYMBOL].astype(str)

        df = self._set_index(df)
        return df

    # ------------------------------------------------------------------ #
    #  财报：接口预留
    # ------------------------------------------------------------------ #

    def load_financial_reports(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """加载财报数据（接口预留，暂未实现）。

        数据存储于 ``fundamentals.duckdb`` 的 ``financial_reports`` 表，
        字段包括：report_date, total_revenue, revenue, net_profit,
        total_assets, total_liabilities, equity, eps, bps, roe,
        gross_margin, net_margin, debt_ratio 等。

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "财报加载接口尚未实现。数据存储于 fundamentals.duckdb → "
            "financial_reports 表，可直接用 duckdb.connect() 查询。"
        )
