"""
combined_fetcher.py — 合并数据获取器（单文件版）

数据源分工：
    BaoStock   → 元数据（股票列表含上市 / 退市日期、交易日历）、日线行情（OHLCV）、每日估值（PE / PB）
    AKShare    → 企业财报（利润表 / 资产负债表 / 现金流量表）

设计原则：
    - 两套内部 fetcher 共享同一 data_root / DuckDB / Parquet 数据栈。
    - 仅实现 CombinedFetcher 实际使用的接口；未使用的接口一律抛出 NotImplementedError。
    - CombinedFetcher 是唯一对外公开的类，_BsFetcher / _AkFetcher 为内部实现细节。

本地存储架构：
    <data_root>/
    ├── meta_data.duckdb          # BaoStock 写入：securities（含 list/delist_date）、trade_calendar
    ├── fundamentals.duckdb       # AKShare 写入：financial_reports；BaoStock 写入：daily_valuation
    └── market_data/              # BaoStock 写入：Hive Parquet OHLCV
        └── market=A_stock/
            └── year=YYYY/
                └── data.parquet
"""

from __future__ import annotations

import json
import random
import time
import traceback
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from threading import Thread
from typing import Any, Generator

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


# ====================================================================
#  辅助：懒加载
# ====================================================================

def _ensure_baostock():
    """懒加载 baostock，未安装则给出清晰提示。"""
    try:
        import baostock as bs
        return bs
    except ImportError:
        raise ImportError("请先安装 baostock:  pip install baostock")


def _ensure_akshare():
    """懒加载 akshare，未安装时给出清晰提示。"""
    try:
        import akshare
        return akshare
    except ImportError:
        raise ImportError("请先安装 akshare:  pip install akshare")


# ====================================================================
#  辅助：BaoStock 会话上下文管理器
# ====================================================================

@contextmanager
def _bs_session() -> Generator:
    """BaoStock 登录/登出上下文管理器。"""
    bs = _ensure_baostock()
    login_result = bs.login()
    if login_result.error_code != '0':
        logger.error("BaoStock 登录失败: {}", login_result.error_msg)
        raise RuntimeError(f"BaoStock 登录失败: {login_result.error_msg}")
    logger.debug("BaoStock 登录成功")
    _patch_bs_socket_timeout(bs, timeout=60)
    try:
        yield bs
    finally:
        bs.logout()
        logger.debug("BaoStock 已登出")


def _patch_bs_socket_timeout(bs, timeout: int = 60) -> None:
    """给 BaoStock 底层 socket 设置超时（秒），防止永久阻塞。"""
    try:
        import baostock.common as _bs_common
        sock = getattr(getattr(_bs_common, "context", None), "default_socket", None)
        if sock is None:
            for attr in ("_client", "client"):
                client = getattr(bs, attr, None)
                if client is not None:
                    if hasattr(client, "settimeout"):
                        sock = client
                    elif hasattr(client, "_socket"):
                        sock = client._socket
                    elif hasattr(client, "socket"):
                        sock = client.socket
                    break
        if sock is not None:
            sock.settimeout(timeout)
            logger.debug("已为 BaoStock socket 设置 {}s 超时", timeout)
        else:
            logger.debug("未找到 BaoStock 底层 socket，跳过超时设置")
    except Exception as exc:
        logger.debug("设置 socket 超时失败（不影响运行）: {}", exc)


def _call_with_timeout(func, args=(), kwargs=None, timeout: int = 120):
    """在子线程中执行 func，超时后抛出 TimeoutError。"""
    kwargs = kwargs or {}
    result = [None]
    exception = [None]

    def _worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    t = Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"{func.__name__} 超过 {timeout}s 未响应，疑似连接断开")
    if exception[0] is not None:
        raise exception[0]
    return result[0]


# ====================================================================
#  辅助：BaoStock 结果集 → pandas DataFrame
# ====================================================================

def _rs_to_dataframe(rs) -> pd.DataFrame:
    """将 BaoStock 分页结果集解析为 DataFrame。"""
    data_list: list[list] = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    return pd.DataFrame(data_list, columns=rs.fields)


# ====================================================================
#  辅助：日期工具
# ====================================================================

def _to_str8(d: str | date | datetime) -> str:
    """任意日期对象 → 'YYYYMMDD' 字符串。"""
    if isinstance(d, (date, datetime)):
        return d.strftime("%Y%m%d")
    s = str(d).replace("-", "").replace("/", "")
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"无法解析为 YYYYMMDD: {d!r}")
    return s


def _to_date(d: str | date | datetime) -> date:
    """任意日期对象 → datetime.date。"""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    s = _to_str8(d)
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _to_dash(d: str | date | datetime) -> str:
    """任意日期对象 → 'YYYY-MM-DD'（BaoStock 要求的日期格式）。"""
    s = _to_str8(d)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


# ====================================================================
#  辅助：股票代码格式转换
# ====================================================================

def _symbol_to_bs(symbol: str) -> str:
    """标准后缀格式 → BaoStock 格式。'600519.SH' → 'sh.600519'"""
    s = symbol.strip().upper()
    if "." in s:
        code, suffix = s.split(".", 1)
        return f"{suffix.lower()}.{code}"
    code = s
    if code.startswith("6"):
        return f"sh.{code}"
    return f"sz.{code}"


def _bs_to_symbol(bs_code: str) -> str:
    """BaoStock 格式 → 标准后缀格式。'sh.600519' → '600519.SH'"""
    parts = bs_code.strip().split(".")
    if len(parts) == 2:
        prefix, code = parts[0].upper(), parts[1]
        return f"{code}.{prefix}"
    return bs_code


def _code_to_suffix(code: str) -> str:
    """6 位纯数字代码 → 标准后缀 symbol。"""
    c = code.strip()
    if c.startswith("6"):
        return f"{c}.SH"
    if c.startswith(("0", "3")):
        return f"{c}.SZ"
    if c.startswith(("4", "8")):
        return f"{c}.BJ"
    return f"{c}.SZ"


def _symbol_to_ak(symbol: str) -> str:
    """标准后缀格式 → akshare 所需的纯 6 位数字代码。"""
    s = symbol.strip().upper()
    for suffix in (".SH", ".SZ", ".BJ"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    for prefix in ("SH", "SZ", "BJ"):
        if s.startswith(prefix) and len(s) == 8:
            return s[len(prefix):]
    return s


# ====================================================================
#  辅助：类型转换
# ====================================================================

def _safe_float_series(series: pd.Series) -> pd.Series:
    """BaoStock 返回的字符串 Series 安全转为 float64。"""
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _safe_float(v) -> float | None:
    """标量值安全转换为 float，空值/NaN 返回 None。"""
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None


# ====================================================================
#  辅助：带指数退避的重试
# ====================================================================

def _retry_with_backoff(
    func,
    *args,
    max_retries: int = 3,
    base_wait: float = 5.0,
    **kwargs,
) -> Any:
    """执行 func，遇到异常时进行指数退避重试，全部失败返回 None。"""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if attempt == max_retries:
                logger.error(
                    "重试 {} 次后仍然失败: {}\n{}",
                    max_retries, exc, traceback.format_exc(),
                )
                return None
            wait = base_wait * (2 ** attempt) + random.uniform(0, 2)
            logger.warning(
                "第 {}/{} 次失败 ({}), 等待 {:.1f}s 后重试…",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    return None  # pragma: no cover


# ====================================================================
#  内部类：_BsFetcher（BaoStock 数据源）
#  仅实现 CombinedFetcher 实际调用的接口；其余接口一律抛出 NotImplementedError
# ====================================================================

class _BsFetcher:
    """BaoStock 数据获取内部类（仅供 CombinedFetcher 使用）。"""

    def __init__(
        self,
        data_root: str | Path = "./cache",
        request_interval: tuple[float, float] = (0.3, 1.0),
        max_retries: int = 3,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.meta_db_path = self.data_root / "meta_data.duckdb"
        self.fund_db_path = self.data_root / "fundamentals.duckdb"
        self.market_dir = self.data_root / "market_data"
        self.market_dir.mkdir(parents=True, exist_ok=True)
        self._progress_path = self.data_root / ".fetch_progress.json"
        self.request_interval = request_interval
        self.max_retries = max_retries

    def _sleep(self) -> None:
        time.sleep(random.uniform(*self.request_interval))

    def _load_progress(self, task_key: str) -> set[str]:
        if not self._progress_path.exists():
            return set()
        data = json.loads(self._progress_path.read_text(encoding="utf-8"))
        return set(data.get(task_key, []))

    def _save_progress(self, task_key: str, done_symbols: set[str]) -> None:
        data: dict[str, list[str]] = {}
        if self._progress_path.exists():
            data = json.loads(self._progress_path.read_text(encoding="utf-8"))
        data[task_key] = sorted(done_symbols)
        self._progress_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _clear_progress(self, task_key: str) -> None:
        if not self._progress_path.exists():
            return
        data = json.loads(self._progress_path.read_text(encoding="utf-8"))
        data.pop(task_key, None)
        self._progress_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ================================================================
    #  1. init_database_schema
    # ================================================================

    def init_database_schema(self) -> None:
        """创建 meta_data.duckdb 和 fundamentals.duckdb 中需要的全部表（幂等）。"""
        with duckdb.connect(str(self.meta_db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS securities (
                    symbol      VARCHAR PRIMARY KEY,
                    market      VARCHAR DEFAULT 'A_stock',
                    asset_type  VARCHAR DEFAULT 'stock',
                    name        VARCHAR,
                    list_date   DATE,
                    delist_date DATE,
                    exchange    VARCHAR
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_calendar (
                    trade_date  DATE NOT NULL,
                    market      VARCHAR NOT NULL DEFAULT 'A_stock',
                    is_open     BOOLEAN NOT NULL DEFAULT TRUE,
                    PRIMARY KEY (trade_date, market)
                )
            """)
        logger.info("meta_data.duckdb 表结构初始化完成")

        with duckdb.connect(str(self.fund_db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS financial_reports (
                    symbol           VARCHAR NOT NULL,
                    report_date      DATE NOT NULL,
                    publish_date     DATE,
                    total_revenue    DOUBLE,
                    revenue          DOUBLE,
                    net_profit       DOUBLE,
                    net_profit_excl  DOUBLE,
                    total_assets     DOUBLE,
                    total_liabilities DOUBLE,
                    equity           DOUBLE,
                    oper_cashflow    DOUBLE,
                    eps              DOUBLE,
                    bps              DOUBLE,
                    roe              DOUBLE,
                    gross_margin     DOUBLE,
                    net_margin       DOUBLE,
                    debt_ratio       DOUBLE,
                    PRIMARY KEY (symbol, report_date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_valuation (
                    symbol       VARCHAR NOT NULL,
                    trade_date   DATE NOT NULL,
                    pe           DOUBLE,
                    pe_ttm       DOUBLE,
                    pb           DOUBLE,
                    ps           DOUBLE,
                    ps_ttm       DOUBLE,
                    total_mv     DOUBLE,
                    circ_mv      DOUBLE,
                    PRIMARY KEY (symbol, trade_date)
                )
            """)
        logger.info("fundamentals.duckdb 表结构初始化完成")

    # ================================================================
    #  2. update_meta_data
    # ================================================================

    def update_meta_data(self) -> None:
        """从 BaoStock 拉取全市场 A 股列表和交易日历，写入 meta_data.duckdb。"""
        with _bs_session() as bs:
            self._update_securities(bs)
            self._sleep()
            self._update_trade_calendar(bs)
        logger.info("元数据更新全部完成")

    def _update_securities(self, bs) -> None:
        logger.info("正在获取 A 股股票列表…")
        rs = bs.query_stock_basic()
        if rs.error_code != '0':
            logger.error("获取股票列表失败: {}", rs.error_msg)
            return
        df = _rs_to_dataframe(rs)
        if df.empty:
            logger.warning("股票列表为空，跳过")
            return
        records: list[tuple] = []
        for _, row in df.iterrows():
            bs_code = str(row.get("code", "")).strip()
            if not bs_code:
                continue
            symbol = _bs_to_symbol(bs_code)
            name = str(row.get("code_name", ""))
            ipo_str = str(row.get("ipoDate", "")).strip()
            out_str = str(row.get("outDate", "")).strip()
            list_date = _to_date(ipo_str) if ipo_str else None
            delist_date = _to_date(out_str) if out_str else None
            exchange = "SSE" if bs_code.startswith("sh") else "SZSE"
            asset_type_raw = str(row.get("type", "1")).strip()
            asset_type = "stock" if asset_type_raw == "1" else "index" if asset_type_raw == "2" else "other"
            records.append((symbol, "A_stock", asset_type, name, list_date, delist_date, exchange))
        if not records:
            logger.warning("解析到 0 条证券记录，跳过")
            return
        with duckdb.connect(str(self.meta_db_path)) as conn:
            conn.execute("DELETE FROM securities WHERE market = 'A_stock'")
            conn.executemany(
                "INSERT INTO securities (symbol, market, asset_type, name, list_date, delist_date, exchange) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                records,
            )
        logger.info("已写入 {} 条 A 股证券记录", len(records))

    def _update_trade_calendar(self, bs) -> None:
        logger.info("正在获取交易日历…")
        today = date.today()
        rs = bs.query_trade_dates(start_date="1990-01-01", end_date=f"{today.year}-12-31")
        if rs.error_code != '0':
            logger.error("获取交易日历失败: {}", rs.error_msg)
            return
        df = _rs_to_dataframe(rs)
        if df.empty:
            logger.warning("交易日历为空，跳过")
            return
        records: list[tuple] = []
        for _, row in df.iterrows():
            cal_date_str = str(row.get("calendar_date", "")).strip()
            is_trading = str(row.get("is_trading_day", "0")).strip()
            if not cal_date_str:
                continue
            try:
                trade_date = _to_date(cal_date_str)
            except ValueError:
                continue
            records.append((trade_date, "A_stock", is_trading == "1"))
        with duckdb.connect(str(self.meta_db_path)) as conn:
            conn.execute("DELETE FROM trade_calendar WHERE market = 'A_stock'")
            conn.executemany(
                "INSERT INTO trade_calendar (trade_date, market, is_open) VALUES (?, ?, ?)",
                records,
            )
        logger.info("已写入 {} 条交易日历记录", len(records))

    # ================================================================
    #  3. fetch_daily_market_data
    # ================================================================

    def fetch_daily_market_data(
        self,
        start_date: str | date = "20100101",
        end_date: str | date | None = None,
        adjustflag: str = "2",
        symbols: list[str] | None = None,
    ) -> None:
        """批量拉取日线行情并以 Hive 分区 Parquet 落盘。"""
        start_d = _to_date(start_date)
        end_d = _to_date(end_date) if end_date else date.today()
        start_bs = _to_dash(start_d)
        end_bs = _to_dash(end_d)

        if symbols is None:
            symbols = self._get_all_symbols()
        if not symbols:
            logger.warning("股票列表为空，请先调用 update_meta_data()")
            return

        task_key = f"bs_market_{_to_str8(start_d)}_{_to_str8(end_d)}_adj{adjustflag}"
        done = self._load_progress(task_key)
        todo = [s for s in symbols if s not in done]
        logger.info("日线行情任务: 共 {} 只，已完成 {}，待下载 {}", len(symbols), len(done), len(todo))

        market_fields = "date,code,open,high,low,close,volume,amount"
        BATCH_SIZE = 200
        batch_counter = 0
        bs_obj = None

        def _reconnect(reason: str = ""):
            nonlocal bs_obj
            if bs_obj is not None:
                try:
                    bs_obj.logout()
                except Exception:
                    pass
            if reason:
                logger.info("重新登录 BaoStock（{}）…", reason)
            bs_mod = _ensure_baostock()
            lr = bs_mod.login()
            if lr.error_code != '0':
                raise RuntimeError(f"BaoStock 重新登录失败: {lr.error_msg}")
            _patch_bs_socket_timeout(bs_mod, timeout=60)
            bs_obj = bs_mod

        def _ensure_session():
            nonlocal batch_counter
            if bs_obj is None or batch_counter >= BATCH_SIZE:
                _reconnect(f"已处理 {batch_counter} 只" if bs_obj else "初始连接")
                batch_counter = 0

        total = len(symbols)
        done_base = len(done)

        try:
            for i, symbol in enumerate(todo, 1):
                _ensure_session()
                bs_code = _symbol_to_bs(symbol)
                logger.info("[{}/{}] 拉取 {} ({}) 日线行情…", done_base + i, total, symbol, bs_code)
                raw_df = None
                for attempt in range(self.max_retries + 1):
                    try:
                        raw_df = _call_with_timeout(
                            self._fetch_single_market,
                            args=(bs_obj, bs_code, start_bs, end_bs, market_fields, adjustflag),
                            timeout=120,
                        )
                        break
                    except Exception as exc:
                        if attempt == self.max_retries:
                            logger.error("{} 重试 {} 次后仍然失败: {}", symbol, self.max_retries, exc)
                            raw_df = None
                        else:
                            wait = 5.0 * (2 ** attempt) + random.uniform(0, 2)
                            logger.warning("{} 第 {}/{} 次失败 ({}), 重连并等待 {:.1f}s…",
                                           symbol, attempt + 1, self.max_retries, exc, wait)
                            _reconnect(f"网络异常重试 {attempt + 1}")
                            time.sleep(wait)

                if raw_df is None or raw_df.empty:
                    logger.warning("{} 无数据或拉取失败，标记完成并跳过", symbol)
                else:
                    df = self._clean_market_df(raw_df, adjustflag)
                    df["year"] = df["trade_date"].dt.year
                    for year, group in df.groupby("year"):
                        self._write_market_partition(group.drop(columns=["year"]), int(year))

                done.add(symbol)
                self._save_progress(task_key, done)
                batch_counter += 1
                self._sleep()
        finally:
            if bs_obj is not None:
                try:
                    bs_obj.logout()
                except Exception:
                    pass

        self._clear_progress(task_key)
        logger.info("日线行情全部下载完成 ✓")

    @staticmethod
    def _fetch_single_market(bs, bs_code: str, start: str, end: str, fields: str, adjustflag: str) -> pd.DataFrame:
        rs = bs.query_history_k_data_plus(bs_code, fields, start_date=start, end_date=end,
                                           frequency="d", adjustflag=adjustflag)
        if rs.error_code != '0':
            raise RuntimeError(f"BaoStock 查询失败 ({bs_code}): {rs.error_msg}")
        return _rs_to_dataframe(rs)

    @staticmethod
    def _clean_market_df(raw_df: pd.DataFrame, adjustflag: str) -> pd.DataFrame:
        """BaoStock 原始行情 DataFrame → 标准格式。"""
        df = raw_df.copy()
        df["symbol"] = df["code"].apply(_bs_to_symbol)
        df["trade_date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = _safe_float_series(df[col])
        df["adjust_flag"] = adjustflag
        keep_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount", "adjust_flag"]
        return df[keep_cols].copy()

    def _write_market_partition(self, df: pd.DataFrame, year: int) -> None:
        """写入 Hive 分区 Parquet（增量去重）。"""
        partition_dir = self.market_dir / "market=A_stock" / f"year={year}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = partition_dir / "data.parquet"
        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        if parquet_path.exists():
            existing = pd.read_parquet(parquet_path)
            existing["trade_date"] = pd.to_datetime(existing["trade_date"])
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["symbol", "trade_date"], keep="last")
        else:
            combined = df
        combined = combined.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), parquet_path, compression="snappy")
        logger.debug("写入分区 year={}: {} 行 → {}", year, len(combined), parquet_path)

    def _get_all_symbols(self, asset_type: str = "stock") -> list[str]:
        try:
            with duckdb.connect(str(self.meta_db_path), read_only=True) as conn:
                rows = conn.execute(
                    "SELECT symbol FROM securities WHERE market = 'A_stock' AND asset_type = ? ORDER BY symbol",
                    [asset_type],
                ).fetchall()
            return [r[0] for r in rows]
        except Exception as exc:
            logger.error("读取股票列表失败: {}", exc)
            return []

    # ================================================================
    #  工具：rebuild_market_progress
    # ================================================================

    def rebuild_market_progress(
        self,
        start_date: str | date = "20160101",
        end_date: str | date | None = None,
        adjustflag: str = "2",
        end_tolerance_days: int = 30,
    ) -> int:
        """扫描已有 Parquet 文件，重建行情下载进度记录。"""
        from datetime import timedelta

        start_d = _to_date(start_date)
        end_d = _to_date(end_date) if end_date else date.today()
        task_key = f"bs_market_{_to_str8(start_d)}_{_to_str8(end_d)}_adj{adjustflag}"

        partition_root = self.market_dir / "market=A_stock"
        if not partition_root.exists():
            logger.warning("market_data 目录不存在，无可恢复的进度")
            return 0

        parquet_files = sorted(partition_root.glob("*/data.parquet"))
        if not parquet_files:
            logger.warning("未找到任何 Parquet 分区文件，无可恢复的进度")
            return 0

        all_frames: list[pd.DataFrame] = []
        for pq_file in parquet_files:
            try:
                table = pq.read_table(pq_file, columns=["symbol", "trade_date"])
                all_frames.append(table.to_pandas())
            except Exception as exc:
                logger.warning("读取 {} 失败，跳过: {}", pq_file, exc)

        if not all_frames:
            logger.warning("所有分区文件均读取失败，无可恢复的进度")
            return 0

        combined = pd.concat(all_frames, ignore_index=True)
        combined["trade_date"] = pd.to_datetime(combined["trade_date"])
        stats = combined.groupby("symbol")["trade_date"].agg(["max", "count"])

        symbol_info: dict[str, tuple[date | None, date | None]] = {}
        try:
            with duckdb.connect(str(self.meta_db_path), read_only=True) as conn:
                rows = conn.execute(
                    "SELECT symbol, list_date, delist_date FROM securities WHERE market = 'A_stock'"
                ).fetchall()
            for sym, ld, dd in rows:
                symbol_info[sym] = (
                    ld.date() if hasattr(ld, "date") else ld,
                    dd.date() if hasattr(dd, "date") else dd,
                )
        except Exception as exc:
            logger.warning("无法读取 meta_data.duckdb 上市/退市信息，跳过该项验证: {}", exc)

        effective_end = min(end_d, date.today())
        done: set[str] = set()
        stale: list[str] = []
        skipped_oor = 0

        for symbol, row in stats.iterrows():
            list_date_sym, delist_date_sym = symbol_info.get(symbol, (None, None))
            if list_date_sym is not None and list_date_sym > end_d:
                done.add(symbol)
                skipped_oor += 1
                continue
            if delist_date_sym is not None and delist_date_sym < start_d:
                done.add(symbol)
                skipped_oor += 1
                continue
            if delist_date_sym is not None and delist_date_sym < effective_end:
                sym_effective_end = delist_date_sym
            else:
                sym_effective_end = effective_end
            sym_end_threshold = pd.Timestamp(sym_effective_end - timedelta(days=end_tolerance_days))
            if row["count"] == 0:
                stale.append(symbol)
            elif row["max"] < sym_end_threshold:
                stale.append(symbol)
                logger.debug("  {} 数据截止 {}，早于阈值 {}，标记为待重拉",
                             symbol, row["max"].date(), sym_end_threshold.date())
            else:
                done.add(symbol)

        for sym, (ld, dd) in symbol_info.items():
            if sym not in stats.index:
                if (ld is not None and ld > end_d) or (dd is not None and dd < start_d):
                    done.add(sym)
                    skipped_oor += 1

        logger.info(
            "已从 {} 个分区恢复进度：task_key={}, 通过验证 {} 只（区间外跳过 {} 只），需重拉 {} 只",
            len(parquet_files), task_key, len(done), skipped_oor, len(stale),
        )
        return len(done)

    # ================================================================
    #  4. fetch_daily_valuation
    # ================================================================

    def fetch_daily_valuation(
        self,
        start_date: str | date = "20200101",
        end_date: str | date | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        """批量拉取日线估值指标（PE/PB），写入 fundamentals.duckdb。"""
        start_d = _to_date(start_date)
        end_d = _to_date(end_date) if end_date else date.today()
        start_bs = _to_dash(start_d)
        end_bs = _to_dash(end_d)

        if symbols is None:
            symbols = self._get_all_symbols()
        if not symbols:
            logger.warning("股票列表为空，请先调用 update_meta_data()")
            return

        task_key = f"bs_valuation_{_to_str8(start_d)}_{_to_str8(end_d)}"
        done = self._load_progress(task_key)
        todo = [s for s in symbols if s not in done]
        logger.info("每日估值任务 ({} ~ {}): 共 {} 只，已完成 {}，待下载 {}",
                    start_d, end_d, len(symbols), len(done), len(todo))

        val_fields = "date,code,peTTM,pbMRQ,psTTM"
        batch: list[tuple] = []
        BATCH_SIZE = 200
        batch_counter = 0
        bs_obj = None

        def _reconnect_val(reason: str = ""):
            nonlocal bs_obj
            if bs_obj is not None:
                try:
                    bs_obj.logout()
                except Exception:
                    pass
            if reason:
                logger.info("估值: 重新登录 BaoStock（{}）…", reason)
            bs_mod = _ensure_baostock()
            lr = bs_mod.login()
            if lr.error_code != '0':
                raise RuntimeError(f"BaoStock 重新登录失败: {lr.error_msg}")
            _patch_bs_socket_timeout(bs_mod, timeout=60)
            bs_obj = bs_mod

        def _ensure_val_session():
            nonlocal batch_counter
            if bs_obj is None or batch_counter >= BATCH_SIZE:
                _reconnect_val(f"已处理 {batch_counter} 只" if bs_obj else "初始连接")
                batch_counter = 0

        total = len(symbols)
        done_base = len(done)

        try:
            for i, symbol in enumerate(todo, 1):
                _ensure_val_session()
                bs_code = _symbol_to_bs(symbol)
                logger.info("[{}/{}] 拉取 {} 估值…", done_base + i, total, symbol)
                raw_df = None
                for attempt in range(self.max_retries + 1):
                    try:
                        raw_df = _call_with_timeout(
                            self._fetch_single_valuation,
                            args=(bs_obj, bs_code, start_bs, end_bs, val_fields),
                            timeout=120,
                        )
                        break
                    except Exception as exc:
                        if attempt == self.max_retries:
                            logger.error("{} 估值重试 {} 次后失败: {}", symbol, self.max_retries, exc)
                            raw_df = None
                        else:
                            wait = 5.0 * (2 ** attempt) + random.uniform(0, 2)
                            logger.warning("{} 估值第 {}/{} 次失败, 重连…", symbol, attempt + 1, self.max_retries)
                            _reconnect_val(f"网络异常重试 {attempt + 1}")
                            time.sleep(wait)

                if raw_df is not None and not raw_df.empty:
                    batch.extend(self._parse_valuation_df(raw_df, symbol))

                done.add(symbol)
                self._save_progress(task_key, done)
                if len(batch) >= 5000:
                    self._insert_valuation_records(batch)
                    batch.clear()
                batch_counter += 1
                self._sleep()
        finally:
            if bs_obj is not None:
                try:
                    bs_obj.logout()
                except Exception:
                    pass

        if batch:
            self._insert_valuation_records(batch)
        self._clear_progress(task_key)
        logger.info("每日估值拉取完成 ✓")

    @staticmethod
    def _fetch_single_valuation(bs, bs_code: str, start: str, end: str, fields: str) -> pd.DataFrame:
        rs = bs.query_history_k_data_plus(bs_code, fields, start_date=start, end_date=end,
                                           frequency="d", adjustflag="3")
        if rs.error_code != '0':
            raise RuntimeError(f"BaoStock 估值查询失败 ({bs_code}): {rs.error_msg}")
        return _rs_to_dataframe(rs)

    @staticmethod
    def _parse_valuation_df(raw_df: pd.DataFrame, symbol: str) -> list[tuple]:
        records: list[tuple] = []
        df = raw_df.copy()
        if df.empty:
            return records
        df["trade_date"] = pd.to_datetime(df["date"])
        for col in ["peTTM", "pbMRQ", "psTTM"]:
            if col in df.columns:
                df[col] = _safe_float_series(df[col])
        for _, row in df.iterrows():
            td = row["trade_date"].date() if hasattr(row["trade_date"], "date") else row["trade_date"]
            records.append((
                symbol, td,
                None, _safe_float(row.get("peTTM")), _safe_float(row.get("pbMRQ")),
                None, _safe_float(row.get("psTTM")), None, None,
            ))
        return records

    def _insert_valuation_records(self, records: list[tuple]) -> None:
        if not records:
            return
        with duckdb.connect(str(self.fund_db_path)) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO daily_valuation "
                "(symbol, trade_date, pe, pe_ttm, pb, ps, ps_ttm, total_mv, circ_mv) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                records,
            )
        logger.info("写入 {} 条估值记录到 fundamentals.duckdb", len(records))

    # ================================================================
    #  5. fetch_daily_market_and_valuation
    # ================================================================

    def fetch_daily_market_and_valuation(
        self,
        start_date: str | date = "20100101",
        end_date: str | date | None = None,
        adjustflag: str = "2",
        symbols: list[str] | None = None,
    ) -> None:
        """一次请求同时拉取日线行情（OHLCV）和估值指标（PE/PB），分别落盘。"""
        start_d = _to_date(start_date)
        end_d = _to_date(end_date) if end_date else date.today()
        start_bs = _to_dash(start_d)
        end_bs = _to_dash(end_d)

        if symbols is None:
            symbols = self._get_all_symbols()
        if not symbols:
            logger.warning("股票列表为空，请先调用 update_meta_data()")
            return

        task_key = f"bs_market_{_to_str8(start_d)}_{_to_str8(end_d)}_adj{adjustflag}"
        done = self._load_progress(task_key)
        todo = [s for s in symbols if s not in done]
        logger.info("行情+估值联合任务: 共 {} 只，已完成 {}，待下载 {}", len(symbols), len(done), len(todo))

        combined_fields = "date,code,open,high,low,close,volume,amount,peTTM,pbMRQ,psTTM"
        BATCH_SIZE = 200
        batch_counter = 0
        bs_obj = None
        val_batch: list[tuple] = []

        def _reconnect(reason: str = ""):
            nonlocal bs_obj
            if bs_obj is not None:
                try:
                    bs_obj.logout()
                except Exception:
                    pass
            if reason:
                logger.info("重新登录 BaoStock（{}）…", reason)
            bs_mod = _ensure_baostock()
            lr = bs_mod.login()
            if lr.error_code != '0':
                raise RuntimeError(f"BaoStock 重新登录失败: {lr.error_msg}")
            _patch_bs_socket_timeout(bs_mod, timeout=60)
            bs_obj = bs_mod

        def _ensure_session():
            nonlocal batch_counter
            if bs_obj is None or batch_counter >= BATCH_SIZE:
                _reconnect(f"已处理 {batch_counter} 只" if bs_obj else "初始连接")
                batch_counter = 0

        total = len(symbols)
        done_base = len(done)

        try:
            for i, symbol in enumerate(todo, 1):
                _ensure_session()
                bs_code = _symbol_to_bs(symbol)
                logger.info("[{}/{}] 拉取 {} 行情+估值…", done_base + i, total, symbol)
                raw_df = None
                for attempt in range(self.max_retries + 1):
                    try:
                        raw_df = _call_with_timeout(
                            self._fetch_single_combined,
                            args=(bs_obj, bs_code, start_bs, end_bs, combined_fields, adjustflag),
                            timeout=120,
                        )
                        break
                    except Exception as exc:
                        if attempt == self.max_retries:
                            logger.error("{} 重试 {} 次后仍然失败: {}", symbol, self.max_retries, exc)
                            raw_df = None
                        else:
                            wait = 5.0 * (2 ** attempt) + random.uniform(0, 2)
                            logger.warning("{} 第 {}/{} 次失败 ({}), 重连并等待 {:.1f}s…",
                                           symbol, attempt + 1, self.max_retries, exc, wait)
                            _reconnect(f"网络异常重试 {attempt + 1}")
                            time.sleep(wait)

                if raw_df is None or raw_df.empty:
                    logger.warning("{} 无数据或拉取失败，标记完成并跳过", symbol)
                else:
                    market_df = self._clean_market_df(raw_df, adjustflag)
                    market_df["year"] = market_df["trade_date"].dt.year
                    for year, group in market_df.groupby("year"):
                        self._write_market_partition(group.drop(columns=["year"]), int(year))
                    val_batch.extend(self._parse_valuation_df(raw_df, symbol))
                    if len(val_batch) >= 5000:
                        self._insert_valuation_records(val_batch)
                        val_batch.clear()

                done.add(symbol)
                self._save_progress(task_key, done)
                batch_counter += 1
                self._sleep()
        finally:
            if bs_obj is not None:
                try:
                    bs_obj.logout()
                except Exception:
                    pass

        if val_batch:
            self._insert_valuation_records(val_batch)
        self._clear_progress(task_key)
        logger.info("行情+估值联合下载完成 ✓")

    @staticmethod
    def _fetch_single_combined(bs, bs_code: str, start: str, end: str, fields: str, adjustflag: str) -> pd.DataFrame:
        rs = bs.query_history_k_data_plus(bs_code, fields, start_date=start, end_date=end,
                                           frequency="d", adjustflag=adjustflag)
        if rs.error_code != '0':
            raise RuntimeError(f"BaoStock 联合查询失败 ({bs_code}): {rs.error_msg}")
        return _rs_to_dataframe(rs)

    # ================================================================
    #  未使用接口（抛出 NotImplementedError）
    # ================================================================

    def fetch_financial_reports(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "BaoStock 的 fetch_financial_reports 在 CombinedFetcher 中未使用。"
            "财报数据由 AKShare 负责，请通过 CombinedFetcher.fetch_financial_reports() 调用。"
        )

    def run_daily_update(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "_BsFetcher.run_daily_update 未实现。"
            "请使用 CombinedFetcher.run_full_update() 或 CombinedFetcher.run_incremental_update()。"
        )


# ====================================================================
#  内部类：_AkFetcher（AKShare 数据源）
#  仅实现财报下载接口；其余接口一律抛出 NotImplementedError
# ====================================================================

class _AkFetcher:
    """AKShare 数据获取内部类（仅供 CombinedFetcher 使用）。"""

    def __init__(
        self,
        data_root: str | Path = "./cache",
        request_interval: tuple[float, float] = (1.5, 3.5),
        max_retries: int = 3,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.meta_db_path = self.data_root / "meta_data.duckdb"
        self.fund_db_path = self.data_root / "fundamentals.duckdb"
        self._progress_path = self.data_root / ".fetch_progress.json"
        self.request_interval = request_interval
        self.max_retries = max_retries

    def _sleep(self) -> None:
        time.sleep(random.uniform(*self.request_interval))

    def _load_progress(self, task_key: str) -> set[str]:
        if not self._progress_path.exists():
            return set()
        data = json.loads(self._progress_path.read_text(encoding="utf-8"))
        return set(data.get(task_key, []))

    def _save_progress(self, task_key: str, done_symbols: set[str]) -> None:
        data: dict[str, list[str]] = {}
        if self._progress_path.exists():
            data = json.loads(self._progress_path.read_text(encoding="utf-8"))
        data[task_key] = sorted(done_symbols)
        self._progress_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _clear_progress(self, task_key: str) -> None:
        if not self._progress_path.exists():
            return
        data = json.loads(self._progress_path.read_text(encoding="utf-8"))
        data.pop(task_key, None)
        self._progress_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_all_symbols(self) -> list[str]:
        try:
            with duckdb.connect(str(self.meta_db_path), read_only=True) as conn:
                rows = conn.execute(
                    "SELECT symbol FROM securities WHERE market = 'A_stock' ORDER BY symbol"
                ).fetchall()
            return [r[0] for r in rows]
        except Exception as exc:
            logger.error("读取股票列表失败: {}", exc)
            return []

    # ================================================================
    #  fetch_financial_reports（已实现）
    # ================================================================

    def fetch_financial_reports(
        self,
        report_period: str = "20241231",
        symbols: list[str] | None = None,
    ) -> None:
        """批量拉取指定报告期的财报数据，写入 fundamentals.duckdb。

        使用 akshare stock_financial_abstract_ths（同花顺财务摘要）接口。

        Parameters
        ----------
        report_period : 报告期，格式 YYYYMMDD，如 '20241231' / '20240630'
        symbols : 股票代码列表；None 则使用全市场
        """
        ak = _ensure_akshare()

        if symbols is None:
            symbols = self._get_all_symbols()
        if not symbols:
            logger.warning("股票列表为空，请先调用 update_meta_data()")
            return

        rp_date = _to_date(report_period)
        task_key = f"financial_{_to_str8(rp_date)}"
        done = self._load_progress(task_key)
        todo = [s for s in symbols if s not in done]
        logger.info("财报拉取任务 (报告期 {}): 共 {} 只，已完成 {}，待下载 {}",
                    rp_date, len(symbols), len(done), len(todo))

        batch_records: list[tuple] = []

        for i, symbol in enumerate(todo, 1):
            code = _symbol_to_ak(symbol)
            logger.info("[{}/{}] 拉取 {} 财报…", i, len(todo), symbol)

            raw_df = _retry_with_backoff(
                self._fetch_single_financial,
                ak, code, symbol,
                max_retries=self.max_retries,
            )

            if raw_df is not None and not raw_df.empty:
                batch_records.extend(self._parse_financial_records(raw_df, symbol, rp_date))

            done.add(symbol)
            self._save_progress(task_key, done)

            if len(batch_records) >= 50:
                self._insert_financial_records(batch_records)
                batch_records.clear()

            self._sleep()

        if batch_records:
            self._insert_financial_records(batch_records)

        self._clear_progress(task_key)
        logger.info("财报拉取全部完成 ✓")

    @staticmethod
    def _fetch_single_financial(ak, code: str, symbol: str) -> pd.DataFrame | None:
        try:
            return ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        except Exception:
            pass
        try:
            return ak.stock_profit_sheet_by_report_em(symbol=code)
        except Exception as exc:
            logger.warning("{} ({}) 财报拉取失败: {}", symbol, code, exc)
            raise

    @staticmethod
    def _parse_financial_records(
        raw_df: pd.DataFrame,
        symbol: str,
        target_date: date,
    ) -> list[tuple]:
        col_map = {
            "报告期": "report_date", "公告日期": "publish_date",
            "营业总收入": "total_revenue", "营业收入": "revenue",
            "净利润": "net_profit", "扣非净利润": "net_profit_excl",
            "总资产": "total_assets", "总负债": "total_liabilities",
            "股东权益合计": "equity", "经营活动产生的现金流量净额": "oper_cashflow",
            "基本每股收益": "eps", "每股净资产": "bps",
            "净资产收益率": "roe", "销售毛利率": "gross_margin",
            "销售净利率": "net_margin", "资产负债率": "debt_ratio",
        }
        df = raw_df.rename(columns=col_map)
        records = []
        for _, row in df.iterrows():
            rd = row.get("report_date")
            if rd is None or pd.isna(rd):
                continue
            try:
                rd = pd.to_datetime(rd).date()
            except Exception:
                continue
            pd_val = row.get("publish_date")
            try:
                pd_val = pd.to_datetime(pd_val).date() if pd_val is not None and not pd.isna(pd_val) else None
            except Exception:
                pd_val = None
            records.append((
                symbol, rd, pd_val,
                _safe_float(row.get("total_revenue")), _safe_float(row.get("revenue")),
                _safe_float(row.get("net_profit")), _safe_float(row.get("net_profit_excl")),
                _safe_float(row.get("total_assets")), _safe_float(row.get("total_liabilities")),
                _safe_float(row.get("equity")), _safe_float(row.get("oper_cashflow")),
                _safe_float(row.get("eps")), _safe_float(row.get("bps")),
                _safe_float(row.get("roe")), _safe_float(row.get("gross_margin")),
                _safe_float(row.get("net_margin")), _safe_float(row.get("debt_ratio")),
            ))
        return records

    def _insert_financial_records(self, records: list[tuple]) -> None:
        if not records:
            return
        with duckdb.connect(str(self.fund_db_path)) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO financial_reports "
                "(symbol, report_date, publish_date, total_revenue, revenue, net_profit, net_profit_excl, "
                "total_assets, total_liabilities, equity, oper_cashflow, eps, bps, roe, gross_margin, "
                "net_margin, debt_ratio) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                records,
            )
        logger.info("写入 {} 条财报记录到 fundamentals.duckdb", len(records))

    # ================================================================
    #  未使用接口（抛出 NotImplementedError）
    # ================================================================

    def init_database_schema(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "AKShare 的 init_database_schema 在 CombinedFetcher 中未使用。"
            "表结构由 BaoStock fetcher 初始化，请通过 CombinedFetcher.init_database_schema() 调用。"
        )

    def update_meta_data(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "AKShare 的 update_meta_data 在 CombinedFetcher 中未使用。"
            "元数据由 BaoStock 提供，请通过 CombinedFetcher.update_meta_data() 调用。"
        )

    def fetch_daily_market_data(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "AKShare 的日线行情下载接口在 CombinedFetcher 中未使用。"
            "日线行情由 BaoStock 负责，请通过 CombinedFetcher.fetch_daily_market_data() 调用。"
        )

    def fetch_daily_valuation(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "AKShare 的每日估值下载接口在 CombinedFetcher 中未使用。"
            "估值数据由 BaoStock 负责，请通过 CombinedFetcher.fetch_daily_valuation() 调用。"
        )

    def run_daily_update(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "_AkFetcher.run_daily_update 未实现。"
            "请使用 CombinedFetcher.run_full_update() 或 CombinedFetcher.run_incremental_update()。"
        )


# ====================================================================
#  CombinedFetcher — 唯一对外公开的类
# ====================================================================

class CombinedFetcher:
    """组合数据获取器：BaoStock（行情）+ AKShare（基本面）。

    Parameters
    ----------
    data_root : 本地数据根目录，两个子 fetcher 共享此路径
    bs_request_interval : BaoStock 请求随机休眠区间（秒）
    ak_request_interval : AKShare 请求随机休眠区间（秒，建议稍长防反爬）
    max_retries : 两个 fetcher 共用的最大重试次数
    """

    def __init__(
        self,
        data_root: str | Path = "./cache",
        bs_request_interval: tuple[float, float] = (0.3, 1.0),
        ak_request_interval: tuple[float, float] = (1.5, 3.5),
        max_retries: int = 3,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()

        # BaoStock：负责元数据 + 日线行情 + 估值
        self._bs = _BsFetcher(
            data_root=data_root,
            request_interval=bs_request_interval,
            max_retries=max_retries,
        )

        # AKShare：负责财报
        self._ak = _AkFetcher(
            data_root=data_root,
            request_interval=ak_request_interval,
            max_retries=max_retries,
        )

        logger.info("CombinedFetcher 初始化完成，数据根目录: {}", self.data_root)

    # ================================================================
    #  Step 1：初始化 DuckDB 表结构
    # ================================================================

    def init_database_schema(self) -> None:
        """初始化 meta_data.duckdb 和 fundamentals.duckdb 表结构（幂等）。

        使用 BaoStock fetcher 的 schema，包含 securities.list_date / delist_date 字段。
        """
        self._bs.init_database_schema()

    # ================================================================
    #  Step 2：元数据（BaoStock）
    # ================================================================

    def update_meta_data(self) -> None:
        """用 BaoStock 更新 A 股票列表（含上市 / 退市日期）和交易日历。

        BaoStock 的 query_stock_basic 接口提供 ipoDate / outDate，
        可用于后续 rebuild_market_progress 中的区间验证。
        """
        self._bs.update_meta_data()

    # ================================================================
    #  Step 3：日线行情（BaoStock）
    # ================================================================

    def fetch_daily_market_data(
        self,
        start_date: str | date = "20160101",
        end_date: str | date | None = None,
        adjustflag: str = "2",
        symbols: list[str] | None = None,
    ) -> None:
        """用 BaoStock 批量拉取日线行情，写入 Hive 分区 Parquet。

        Parameters
        ----------
        start_date : 起始日期
        end_date : 截止日期，默认今日
        adjustflag : '1'=后复权, '2'=前复权（默认）, '3'=不复权
        symbols : 股票列表；None 则读取 meta_data.duckdb 全量
        """
        self._bs.fetch_daily_market_data(
            start_date=start_date,
            end_date=end_date,
            adjustflag=adjustflag,
            symbols=symbols,
        )

    def rebuild_market_progress(
        self,
        start_date: str | date = "20160101",
        end_date: str | date | None = None,
        adjustflag: str = "2",
        end_tolerance_days: int = 30,
    ) -> int:
        """扫描现有 Parquet、结合上市 / 退市日期，重建行情下载进度。

        进度文件丢失或程序被强制中断后，可调用此方法恢复断点续传状态。

        Returns
        -------
        int : 通过验证并写入进度文件的 symbol 数量
        """
        return self._bs.rebuild_market_progress(
            start_date=start_date,
            end_date=end_date,
            adjustflag=adjustflag,
            end_tolerance_days=end_tolerance_days,
        )

    # ================================================================
    #  Step 4：每日估值（BaoStock，与行情共用 API）
    # ================================================================

    def fetch_daily_valuation(
        self,
        start_date: str | date = "20100101",
        end_date: str | date | None = None,
        adjustflag: str = "2",
        symbols: list[str] | None = None,
    ) -> None:
        """用 BaoStock 单独拉取日线估值指标，写入 fundamentals.duckdb。

        通常不需要单独调用此方法——推荐使用 fetch_daily_market_and_valuation
        一次请求同时获取行情和估值。此方法保留用于仅需补全估值数据的场景。

        Parameters
        ----------
        start_date : 起始日期
        end_date : 截止日期，默认今日
        adjustflag : 复权标志（仅影响 OHLCV，估值字段不受影响）
        symbols : None 则读取 meta_data.duckdb 全量
        """
        self._bs.fetch_daily_valuation(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )

    def fetch_daily_market_and_valuation(
        self,
        start_date: str | date = "20100101",
        end_date: str | date | None = None,
        adjustflag: str = "2",
        symbols: list[str] | None = None,
    ) -> None:
        """用 BaoStock 一次请求同时拉取日线行情（OHLCV）和估值（PE/PB），分别落盘。

        行情写入 market_data/ Hive Parquet，估值写入 fundamentals.duckdb。
        比分别调用 fetch_daily_market_data + fetch_daily_valuation 减少一半 API 调用。

        Parameters
        ----------
        start_date : 起始日期
        end_date : 截止日期，默认今日
        adjustflag : '1'=后复权, '2'=前复权（默认）, '3'=不复权
        symbols : None 则读取 meta_data.duckdb 全量
        """
        self._bs.fetch_daily_market_and_valuation(
            start_date=start_date,
            end_date=end_date,
            adjustflag=adjustflag,
            symbols=symbols,
        )

    # ================================================================
    #  Step 5：财报数据（AKShare）
    # ================================================================

    def fetch_financial_reports(
        self,
        report_period: str | date | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        """用 AKShare（同花顺财务摘要接口）拉取财报，写入 fundamentals.duckdb。

        AKShare 的 stock_financial_abstract_ths 按报告期返回全历史数据，
        包含利润表 / 资产负债表 / 现金流量表核心指标。

        Parameters
        ----------
        report_period : 报告期（仅用于断点续传 key 命名），None 则自动推断最近季报期
        symbols : None 则读取 meta_data.duckdb 全量
        """
        if report_period is None:
            today = date.today()
            quarter_ends = [
                date(today.year - 1, 12, 31),
                date(today.year, 3, 31),
                date(today.year, 6, 30),
                date(today.year, 9, 30),
                date(today.year, 12, 31),
            ]
            recent = max(q for q in quarter_ends if q <= today)
            report_period = _to_str8(recent)

        self._ak.fetch_financial_reports(
            report_period=str(report_period),
            symbols=symbols,
        )

    # ================================================================
    #  便捷方法：日常增量更新
    # ================================================================

    def run_incremental_update(
        self,
        lookback_days: int = 7,
        adjustflag: str = "2",
        update_meta: bool = True,
        fetch_valuation: bool = True,
        fetch_financial: bool = False,
    ) -> None:
        """日常盘后增量更新：仅拉取最近 N 天的数据，避免重走全量历史。

        与 run_full_update 的区别：
            - 行情 start_date = 今日 - lookback_days（task_key 基于日期，量小跑快）
            - 估值 trade_date 固定为今日（task_key 稳定，中断可续传）
            - 财报默认不拉取（季报期附近才需要，可手动传 fetch_financial=True）
            - 不重跑历史 Parquet 扫描

        Parameters
        ----------
        lookback_days : 行情回溯天数，默认 7（覆盖节假日及数据延迟）
        adjustflag : 复权方式，'2'=前复权（默认）
        update_meta : 是否更新股票列表 / 交易日历（新股上市时需要），默认 True
        fetch_valuation : 是否拉取今日估值，默认 True
        fetch_financial : 是否拉取最近季报，默认 False（季报期附近手动开启）
        """
        from datetime import timedelta

        today = date.today()
        start_date = today - timedelta(days=lookback_days)

        logger.info("=" * 60)
        logger.info("CombinedFetcher 开始日常增量更新（回溯 {} 天）…", lookback_days)
        logger.info("  行情区间: {} ~ {}", start_date, today)
        logger.info("=" * 60)

        # Step 1: 初始化表结构（幂等，耗时极低）
        self.init_database_schema()

        # Step 2: 元数据（增量更新新上市股票 + 交易日历）
        if update_meta:
            logger.info("[1/4] 更新元数据（BaoStock）…")
            self.update_meta_data()
        else:
            logger.info("[1/4] 跳过元数据更新（update_meta=False）")

        # Step 3+4: 近期行情 + 估值（一次 BaoStock 请求，task_key 含日期范围）
        if fetch_valuation:
            logger.info("[2/4] 拉取近期行情+估值（BaoStock，{} ~ {}）…", start_date, today)
            self.fetch_daily_market_and_valuation(
                start_date=start_date,
                end_date=today,
                adjustflag=adjustflag,
            )
        else:
            logger.info("[2/4] 拉取近期日线行情（BaoStock，{} ~ {}）…", start_date, today)
            self.fetch_daily_market_data(
                start_date=start_date,
                end_date=today,
                adjustflag=adjustflag,
            )
            logger.info("[3/4] 跳过估值更新（fetch_valuation=False）")

        # Step 5: 最近季报（默认跳过；季报发布期附近手动开启）
        if fetch_financial:
            logger.info("[4/4] 拉取最近季报（AKShare）…")
            self.fetch_financial_reports()
        else:
            logger.info("[4/4] 跳过财报更新（fetch_financial=False）")

        logger.info("=" * 60)
        logger.info("CombinedFetcher 日常增量更新完成 ✓")
        logger.info("=" * 60)

    # ================================================================
    #  便捷方法：一键全量更新
    # ================================================================

    def run_full_update(
        self,
        start_date: str | date = "20160101",
        end_date: str | date | None = None,
        adjustflag: str = "2",
        fetch_valuation: bool = True,
        fetch_financial: bool = True,
    ) -> None:
        """一键执行完整的盘后数据更新流程。

        执行顺序：
            1. 初始化 DuckDB 表结构（幂等）
            2. 更新元数据             （BaoStock：股票列表 + 交易日历）
            3+4. 拉取日线行情 + 估值  （BaoStock：OHLCV + PE/PB，一次请求）
            5. 拉取财报数据           （AKShare ：利润表 / 资产负债表 / 现金流）

        Parameters
        ----------
        start_date : 行情起始日期
        end_date : 行情截止日期，默认今日
        adjustflag : 复权方式，'2'=前复权（默认）
        fetch_valuation : 是否拉取每日估值
        fetch_financial : 是否拉取财报
        """
        logger.info("=" * 60)
        logger.info("CombinedFetcher 开始全量更新…")
        logger.info("  行情数据源：BaoStock")
        logger.info("  基本面数据源：AKShare")
        logger.info("=" * 60)

        # Step 1: 初始化 DB 表结构
        logger.info("[1/5] 初始化数据库表结构…")
        self.init_database_schema()

        # Step 2: 元数据（股票列表 + 交易日历）
        logger.info("[2/5] 更新元数据（BaoStock）…")
        self.update_meta_data()

        # Step 3+4: 日线行情 + 估值（一次 BaoStock 请求同时获取）
        if fetch_valuation:
            logger.info("[3/5] 拉取日线行情+估值（BaoStock）…")
            self.fetch_daily_market_and_valuation(
                start_date=start_date,
                end_date=end_date,
                adjustflag=adjustflag,
            )
        else:
            logger.info("[3/5] 拉取日线行情（BaoStock）…")
            self.fetch_daily_market_data(
                start_date=start_date,
                end_date=end_date,
                adjustflag=adjustflag,
            )
            logger.info("[4/5] 跳过每日估值（fetch_valuation=False）")

        # Step 5: 财报
        if fetch_financial:
            logger.info("[5/5] 拉取财报数据（AKShare）…")
            self.fetch_financial_reports()
        else:
            logger.info("[5/5] 跳过财报数据（fetch_financial=False）")

        logger.info("=" * 60)
        logger.info("CombinedFetcher 全量更新完成 ✓")
        logger.info("=" * 60)


# ====================================================================
#  入口：直接运行可执行盘后全量更新
# ====================================================================

if __name__ == "__main__":
    fetcher = CombinedFetcher(data_root="./cache")
    fetcher.run_full_update(start_date="20160101", end_date="20260101")
