"""本地文件数据加载器 — 支持 CSV / Parquet / SQLite。"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from data.base import DataLoader
from data.schema import Col


class LocalLoader(DataLoader):
    """从本地文件加载数据。

    Parameters
    ----------
    market_path : CSV / Parquet 文件路径，或 SQLite 数据库路径
    fundamental_path : 基本面数据文件路径（可选）
    market_table : 当数据源为 SQLite 时的行情表名
    fundamental_table : 当数据源为 SQLite 时的基本面表名
    column_mapping : 列名映射字典 {源列名: 标准列名}
    """

    def __init__(
        self,
        market_path: str | Path | None = None,
        fundamental_path: str | Path | None = None,
        market_table: str = "market",
        fundamental_table: str = "fundamental",
        column_mapping: dict[str, str] | None = None,
    ) -> None:
        self.market_path = Path(market_path) if market_path else None
        self.fundamental_path = Path(fundamental_path) if fundamental_path else None
        self.market_table = market_table
        self.fundamental_table = fundamental_table
        self.column_mapping = column_mapping

    # ------------------------------------------------------------------ #
    #  内部：根据后缀读取文件
    # ------------------------------------------------------------------ #

    def _read_file(self, path: Path, table: str | None = None) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            logger.info("读取 CSV: {}", path)
            df = pd.read_csv(path)
        elif suffix in (".parquet", ".pq"):
            logger.info("读取 Parquet: {}", path)
            df = pd.read_parquet(path)
        elif suffix in (".db", ".sqlite", ".sqlite3"):
            logger.info("读取 SQLite: {} 表={}", path, table)
            conn = sqlite3.connect(str(path))
            try:
                df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
            finally:
                conn.close()
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
        return df

    # ------------------------------------------------------------------ #
    #  DataLoader 接口实现
    # ------------------------------------------------------------------ #

    def load_market_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        if self.market_path is None:
            raise ValueError("未指定 market_path")

        df = self._read_file(self.market_path, self.market_table)
        df = self._standardize(df, self.column_mapping)
        df = self._filter(df, symbols, start, end)
        df = self._set_index(df)

        missing = set(Col.market_required()) - set(df.columns) - set(df.index.names)
        if missing:
            logger.warning("行情数据缺少列: {}", missing)

        return df

    def load_fundamental_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        path = self.fundamental_path or self.market_path
        if path is None:
            raise ValueError("未指定 fundamental_path")

        df = self._read_file(path, self.fundamental_table)
        df = self._standardize(df, self.column_mapping)
        df = self._filter(df, symbols, start, end)
        df = self._set_index(df)
        return df
