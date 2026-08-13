"""读取按股票拆分的数据、指数成分股和基准数据。"""

import csv
from datetime import date
from functools import lru_cache
from pathlib import Path
import pandas as pd


DATE_COLUMN_CANDIDATES = ("date", "trade_date")

FIELD_ALIASES = {
    "gross_margin": ("grossprofit_margin", "gross_margin_calc"),
    "operating_cash_flow": (
        "operating_cashflow",
        "n_cashflow_act",
        "im_net_cashflow_oper_act",
    ),
    "share_factor": ("adj_factor",),
    "sector": ("industry_1", "stock_basic_industry"),
    "SPY_close": ("hs300_close",),
    "QQQ_close": ("zz500_close",),
    "HS300_close": ("hs300_close",),
    "ZZ500_close": ("zz500_close",),
    "ZZ1000_close": ("zz1000_close",),
}

DERIVED_FIELD_SOURCES = {
    "gross_profit": ("revenue", "cost_revenue"),
    "gross_margin": ("gross_profit", "revenue", "cost_revenue"),
    "vwap": ("amount", "volume", "close", "adj_close"),
    "non_current_debt": (
        "lt_borr",
        "bond_payable",
        "lt_payable",
        "lease_liab",
        "non_cur_liab_due_1y",
    ),
}

BENCHMARK_COLUMNS = {
    "hs300_close": "HS300",
    "zz500_close": "ZZ500",
    "zz1000_close": "ZZ1000",
    "SPY_close": "SPY",
    "QQQ_close": "QQQ",
}

NON_STOCK_CSV_FILES = {"constituents_daily.csv", "index_members_rebalance.csv"}

# 2015-06-30 至 2026-06-30 的沪深300、中证500、中证1000历史股票池中，
# 经交易所/上市公司公告与 ticker_code_history_v1 交叉核对后共有以下四次
# “同一上市证券直接更换代码”。吸收合并、换股退市和代码复用不属于别名，
# 不得在这里映射，否则会把不同证券主体的行情错误拼接。
# 元组字段为 (旧代码, 现行代码, 新代码启用日)。加载器按现行代码归一，
# 启用日用于审计，运行时不依赖 tmp_data 或外部映射文件。
SECURITY_CODE_CHANGE_RULES = (
    ("000022.SZ", "001872.SZ", date(2018, 12, 26)),
    ("601313.SH", "601360.SH", date(2018, 2, 28)),
    ("000043.SZ", "001914.SZ", date(2019, 12, 16)),
    ("300114.SZ", "302132.SZ", date(2025, 2, 17)),
)

SECURITY_CODE_ALIASES = {
    legacy: current for legacy, current, _effective_date in SECURITY_CODE_CHANGE_RULES
}


def _read_header(csv_file: Path) -> set[str]:
    """读取 CSV 表头。"""
    return set(pd.read_csv(csv_file, nrows=0).columns)


def _canonical_symbol(file_symbol: str, stock_basic_symbol: str) -> str:
    """根据 cache 内的现行代码，把历史文件代码归一到同一证券。"""
    current = stock_basic_symbol.strip()
    if not current:
        return file_symbol
    if "." in current:
        return current
    suffix = file_symbol.rsplit(".", 1)[1] if "." in file_symbol else ""
    return f"{current}.{suffix}" if suffix else current


@lru_cache(maxsize=16)
def _load_symbol_aliases_cached(directory: Path) -> tuple[tuple[str, str], ...]:
    aliases: dict[str, str] = dict(SECURITY_CODE_ALIASES)
    aliases.update({symbol: symbol for symbol in SECURITY_CODE_ALIASES.values()})
    for csv_file in sorted(directory.glob("*.csv")):
        if csv_file.name in NON_STOCK_CSV_FILES:
            continue
        canonical = csv_file.stem
        with csv_file.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            header = next(reader, [])
            if "stock_basic_symbol" in header:
                first_row = next(reader, [])
                column = header.index("stock_basic_symbol")
                if column < len(first_row):
                    canonical = _canonical_symbol(csv_file.stem, first_row[column])
        canonical = SECURITY_CODE_ALIASES.get(csv_file.stem, canonical)
        aliases[csv_file.stem] = canonical
        aliases.setdefault(canonical, canonical)
    return tuple(sorted(aliases.items()))


def load_symbol_aliases(data_dir: str | Path) -> dict[str, str]:
    """从股票宽表自身推导“历史代码→现行代码”，不依赖外部映射文件。"""
    directory = Path(data_dir).expanduser().resolve()
    return dict(_load_symbol_aliases_cached(directory))


def resolve_date_column(columns: set[str], csv_file: Path) -> str:
    """返回数据文件实际使用的日期列名。"""
    for column in DATE_COLUMN_CANDIDATES:
        if column in columns:
            return column
    raise ValueError(f"{csv_file} 缺少日期列，支持字段: {DATE_COLUMN_CANDIDATES}")


def get_source_columns_for_field(field: str, available: set[str]) -> list[str]:
    """把统一字段名解析为当前 CSV 需要读取的源字段。"""
    if field == "date":
        return []

    direct_fields = []
    direct_fields.extend(FIELD_ALIASES.get(field, ()))
    direct_fields.append(field)
    for candidate in direct_fields:
        if candidate in available:
            return [candidate]

    if field in DERIVED_FIELD_SOURCES:
        sources = []
        for column in DERIVED_FIELD_SOURCES[field]:
            if column in available:
                sources.append(column)
        if sources:
            return sources

    return []


def resolve_source_columns(
    requested_columns: list[str] | None,
    available: set[str],
    date_column: str,
    csv_file: Path,
) -> list[str] | None:
    """构造 ``read_csv`` 的 usecols，并在字段缺失时给出明确错误。"""
    if requested_columns is None:
        return None

    source_columns = [date_column]
    for field in dict.fromkeys(requested_columns):
        if field == "date":
            continue
        resolved = get_source_columns_for_field(field, available)
        if not resolved:
            raise ValueError(f"{csv_file} 缺少字段: {field}")
        source_columns.extend(resolved)
    return list(dict.fromkeys(source_columns))


def normalize_fields(frame: pd.DataFrame, requested_columns: list[str] | None) -> pd.DataFrame:
    """把 A 股源字段规范为因子代码使用的统一字段名。"""
    if "trade_date" in frame.columns:
        if "date" in frame.columns:
            frame = frame.drop(columns=["trade_date"])
        else:
            frame = frame.rename(columns={"trade_date": "date"})

    target_columns = set(requested_columns) if requested_columns is not None else None
    for target, candidates in FIELD_ALIASES.items():
        if target_columns is not None and target not in target_columns:
            continue
        for candidate in candidates:
            if candidate in frame.columns:
                frame[target] = frame[candidate]
                break

    for target, sources in DERIVED_FIELD_SOURCES.items():
        if target in frame.columns:
            continue

        if target == "gross_profit":
            revenue_column, cost_column = sources
            if {revenue_column, cost_column}.issubset(frame.columns):
                frame[target] = frame[revenue_column] - frame[cost_column]

        elif target == "gross_margin":
            gross_profit_column = sources[0]
            revenue_column = sources[1]
            if {gross_profit_column, revenue_column}.issubset(frame.columns):
                frame[target] = (
                    frame[gross_profit_column] / frame[revenue_column].replace(0, pd.NA)
                )

        elif target == "vwap":
            amount_column, volume_column, close_column, adj_close_column = sources
            if set(sources).issubset(frame.columns):
                raw_vwap = (
                    frame[amount_column]
                    * 10
                    / frame[volume_column].where(frame[volume_column] != 0)
                )
                frame[target] = (
                    raw_vwap
                    * frame[adj_close_column]
                    / frame[close_column].where(frame[close_column] != 0)
                )

        elif target == "non_current_debt":
            debt_sources = []
            for column in sources:
                if column in frame.columns:
                    debt_sources.append(column)
            if debt_sources:
                frame[target] = frame[debt_sources].fillna(0).sum(axis=1, min_count=1)

    if requested_columns is None:
        return frame

    output_columns = ["date"]
    for column in dict.fromkeys(requested_columns):
        if column == "date":
            continue
        output_columns.append(column)
    return frame[output_columns]


def load_data(
    data_dir: str | Path,
    symbols: list[str] | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """返回 (date, symbol) MultiIndex 数据"""
    data_dir = Path(data_dir).expanduser().resolve()
    symbol_aliases = load_symbol_aliases(data_dir)
    all_csv_files = sorted(
        csv_file
        for csv_file in data_dir.glob("*.csv")
        if csv_file.name not in NON_STOCK_CSV_FILES
    )

    if symbols is None:
        csv_files = all_csv_files
    else:
        requested = {symbol_aliases.get(symbol, symbol) for symbol in symbols}
        csv_files = [
            csv_file
            for csv_file in all_csv_files
            if symbol_aliases.get(csv_file.stem, csv_file.stem) in requested
        ]

    if csv_files:
        available_columns = _read_header(csv_files[0])
        date_column = resolve_date_column(available_columns, csv_files[0])
        usecols = resolve_source_columns(
            columns,
            available_columns,
            date_column,
            csv_files[0],
        )

    frames = []
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file, usecols=usecols, parse_dates=[date_column])
        frame = normalize_fields(frame, columns)
        canonical = symbol_aliases.get(csv_file.stem, csv_file.stem)
        frame["symbol"] = canonical
        frame["_source_priority"] = int(csv_file.stem != canonical)
        if start is not None:
            frame = frame[frame["date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame["date"] <= pd.Timestamp(end)]
        frames.append(frame)

    if frames:
        data = pd.concat(frames, ignore_index=True)
        data = (
            data.sort_values(["date", "symbol", "_source_priority"])
            .drop_duplicates(["date", "symbol"], keep="first")
            .drop(columns="_source_priority")
        )
        data = data.set_index(["date", "symbol"]).sort_index()
    else:
        index = pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
        data = pd.DataFrame(index=index)

    data.attrs["symbol_aliases"] = symbol_aliases
    return data


def filter_by_constituent(
    data: pd.DataFrame,
    constituents: pd.DataFrame,
) -> pd.DataFrame:
    """按每日指数成分股过滤 (date, symbol) MultiIndex 数据。"""
    if data.empty or constituents.empty:
        return data.iloc[0:0]
    if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
        raise ValueError("data 必须使用 (date, symbol) MultiIndex")

    member_mask = constituents.astype(bool).stack()
    member_index = member_mask[member_mask].index
    return data.loc[data.index.isin(member_index)].sort_index()


def align_constituents_to_data(
    constituents: pd.DataFrame,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """将成分股矩阵前向填充到行情交易日。"""
    if data.empty:
        return constituents

    data_dates = pd.DatetimeIndex(
        data.index.get_level_values("date").unique()
    ).sort_values()
    target_dates = data_dates[data_dates >= constituents.index.min()]
    if target_dates.empty:
        raise ValueError("成分股日期与行情日期没有交集")

    constituents = (
        constituents.reindex(constituents.index.union(target_dates))
        .astype("boolean")
        .sort_index()
        .ffill()
        .reindex(target_dates)
        .fillna(False)
        .astype(bool)
    )
    constituents.index.name = "date"
    constituents.columns.name = "symbol"
    return constituents


def _truthy(series: pd.Series) -> pd.Series:
    """将 tushare 状态字段转换为布尔值。"""
    return series.astype(str).str.lower().isin({"t", "true", "1", "y", "yes"})


def filter_by_security_status(
    data: pd.DataFrame,
    security_status: pd.DataFrame,
) -> pd.DataFrame:
    """剔除每日 ST、停牌股票。"""
    if data.empty or security_status.empty:
        return data
    if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
        raise ValueError("data 必须使用 (date, symbol) MultiIndex")

    excluded = security_status.astype(bool).stack()
    excluded_index = excluded[excluded].index
    return data.loc[~data.index.isin(excluded_index)].sort_index()


class DataLoader:
    """管理 Runner 所需的行情、因子结果和基准数据。"""

    def __init__(
        self,
        data_dir: str | Path,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
        columns: list[str] | None = None,
        factor_dir: str | Path | None = None,
        factor_columns: list[str] | None = None,
        constituents_path: str | Path | None = None,
        constituent_index: str | None = None,
        security_status_path: str | Path | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.symbols = symbols
        self.start = start
        self.end = end
        self.columns = columns
        self.factor_dir = factor_dir
        self.factor_columns = factor_columns
        self.constituents_path = constituents_path
        self.constituent_index = constituent_index
        self.security_status_path = security_status_path
        if (constituents_path is None) != (constituent_index is None):
            raise ValueError("constituents_path 和 constituent_index 必须同时为 None 或同时不为 None")
        self.data = pd.DataFrame()
        self.factor_result = pd.DataFrame()
        self.benchmark = pd.DataFrame()
        self.constituents = pd.DataFrame()
        self.security_status = pd.DataFrame()
        self.symbol_aliases: dict[str, str] = {}
        self.trading_calendar = pd.DatetimeIndex([], name="date")

    def load_basic(self) -> pd.DataFrame:
        """加载基础行情数据和可选成分股。"""
        basic_data = load_data(
            data_dir=self.data_dir,
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            columns=self.columns,
        )
        self.symbol_aliases = basic_data.attrs.get("symbol_aliases", {})
        self.trading_calendar = pd.DatetimeIndex(
            basic_data.index.get_level_values("date").unique(),
            name="date",
        ).sort_values()
        return basic_data

    def load_constituents(self) -> pd.DataFrame:
        """加载 ``date,tickers`` 格式的逐日成分股数据。"""

        if self.constituents_path is None:
            return pd.DataFrame()
        constituents_path = Path(self.constituents_path)
        if not constituents_path.exists():
            raise FileNotFoundError(f"成分股文件不存在: {constituents_path}")

        constituents_daily = pd.read_csv(
            constituents_path,
            dtype={"tickers": str},
            usecols=["date", "tickers"],
            parse_dates=["date"],
        )
        if self.start is not None:
            constituents_daily = constituents_daily[
                constituents_daily["date"] >= pd.Timestamp(self.start)
            ]
        if self.end is not None:
            constituents_daily = constituents_daily[
                constituents_daily["date"] <= pd.Timestamp(self.end)
            ]
        if constituents_daily.empty:
            raise ValueError(
                f"在 {self.start} 到 {self.end} 期间没有找到 "
                f"{self.constituent_index} 的逐日成分股数据"
            )

        constituents = (
            constituents_daily.assign(
                symbol=constituents_daily["tickers"].fillna("").str.split(",")
            )
            .explode("symbol")[["date", "symbol"]]
        )
        constituents["symbol"] = constituents["symbol"].str.strip()
        constituents = constituents[constituents["symbol"].ne("")]
        constituents["symbol"] = constituents["symbol"].map(
            lambda symbol: self.symbol_aliases.get(symbol, symbol)
        )
        if constituents.empty:
            raise ValueError(f"{constituents_path} 没有可用成分股数据")

        constituents["is_member"] = True
        constituents = (
            constituents.pivot_table(
                index="date",
                columns="symbol",
                values="is_member",
                aggfunc="last",
                fill_value=False,
            )
            .astype(bool)
            .sort_index()
            .sort_index(axis=1)
        )
        constituents.index.name = "date"
        constituents.columns.name = "symbol"

        return align_constituents_to_data(constituents, self.data)

    def build_suspended_status_table(
        self,
        status: pd.DataFrame,
    ) -> pd.DataFrame:
        """将逐日停牌记录构造为与行情数据对齐的布尔表。"""
        data_dates = pd.DatetimeIndex(
            self.data.index.get_level_values("date").unique()
        ).sort_values()
        data_symbols = pd.Index(self.data.index.get_level_values("symbol").unique())

        suspended = status[["ticker", "trade_date", "is_suspended"]].copy()
        suspended["date"] = pd.to_datetime(
            suspended["trade_date"],
            errors="coerce",
        )
        suspended["is_excluded"] = _truthy(suspended["is_suspended"])
        suspended = suspended.dropna(subset=["date"])

        suspended_status = suspended.pivot_table(
            index="date",
            columns="ticker",
            values="is_excluded",
            aggfunc="last",
        )
        suspended_status = (
            suspended_status.reindex(index=data_dates, columns=data_symbols)
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        suspended_status.index.name = "date"
        suspended_status.columns.name = "symbol"
        return suspended_status

    def build_st_status_table(
        self,
        status: pd.DataFrame,
    ) -> pd.DataFrame:
        """将 ST 状态区间构造为与行情数据对齐的布尔表。"""
        data_dates = pd.DatetimeIndex(
            self.data.index.get_level_values("date").unique()
        ).sort_values()
        data_symbols = pd.Index(self.data.index.get_level_values("symbol").unique())

        intervals = status.copy()
        intervals["start"] = pd.to_datetime(
            intervals["start_date"],
            errors="coerce",
        )
        intervals["end"] = pd.to_datetime(
            intervals["end_date"],
            errors="coerce",
        ).fillna(data_dates[-1])
        intervals["is_excluded"] = _truthy(intervals["is_st"])
        intervals = intervals.dropna(subset=["start"])

        records = []
        for row in intervals.itertuples(index=False):
            left = data_dates.searchsorted(row.start, side="left")
            right = data_dates.searchsorted(row.end, side="right")
            dates = data_dates[left:right]
            if dates.empty:
                continue
            records.append(pd.DataFrame({
                "date": dates,
                "ticker": row.ticker,
                "is_excluded": row.is_excluded,
            }))

        if records:
            normalized = pd.concat(records, ignore_index=True).dropna(subset=["date"])
        else:
            normalized = pd.DataFrame(columns=["date", "ticker", "is_excluded"])

        st_status = normalized.pivot_table(
            index="date",
            columns="ticker",
            values="is_excluded",
            aggfunc="last",
        )
        st_status = (
            st_status.reindex(index=data_dates, columns=data_symbols)
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        st_status.index.name = "date"
        st_status.columns.name = "symbol"
        return st_status

    def load_security_status(self) -> pd.DataFrame:
        """读取停牌和 ST 状态，并对齐到行情日期。"""
        if self.data.empty or self.security_status_path is None:
            return pd.DataFrame()

        security_status_path = Path(self.security_status_path)
        if not security_status_path.exists():
            raise FileNotFoundError(f"证券状态文件不存在: {security_status_path}")

        columns = [
            "ticker",
            "trade_date",
            "start_date",
            "end_date",
            "is_suspended",
            "is_st",
        ]
        status = pd.read_csv(
            security_status_path,
            usecols=columns,
            dtype=str,
        )
        status["ticker"] = status["ticker"].map(
            lambda symbol: self.symbol_aliases.get(symbol, symbol)
        )

        data_symbols = pd.Index(self.data.index.get_level_values("symbol").unique())
        status = status[
            status["ticker"].isin(data_symbols)
            & (status["is_suspended"].notna() | status["is_st"].notna())
        ].copy()

        suspended_status = self.build_suspended_status_table(
            status[status["is_suspended"].notna()]
        )
        st_status = self.build_st_status_table(
            status[status["is_st"].notna()]
        )
        return suspended_status | st_status

    def load_factor_result(self) -> pd.DataFrame:
        """加载已保存的因子宽表，并合并到基础行情数据。"""

        factor_result = load_data(
            data_dir=self.factor_dir,
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            columns=self.factor_columns,
        )
        if not self.data.empty:
            self.data = self.data.join(factor_result, how="left")
        return factor_result

    def load_benchmark(self) -> pd.DataFrame:
        """提取回测期间可用的指数基准收盘价。"""
        if self.data.empty:
            return pd.DataFrame()
        source_symbol = self.data.index.get_level_values("symbol")[0]
        data_dir = Path(self.data_dir).expanduser().resolve()
        source_file = data_dir / f"{source_symbol}.csv"
        if not source_file.exists():
            source_file = next(
                data_dir / f"{alias}.csv"
                for alias, canonical in self.symbol_aliases.items()
                if canonical == source_symbol and (data_dir / f"{alias}.csv").exists()
            )
        available_columns = _read_header(source_file)
        benchmark_columns = [
            column for column in BENCHMARK_COLUMNS
            if column in available_columns
        ]
        if not benchmark_columns:
            return pd.DataFrame()
        benchmark_data = load_data(
            data_dir=self.data_dir,
            symbols=[source_symbol],
            start=self.start,
            end=self.end,
            columns=benchmark_columns,
        )
        benchmark = benchmark_data.xs(source_symbol, level="symbol").rename(
            columns=BENCHMARK_COLUMNS
        )
        benchmark = benchmark[[BENCHMARK_COLUMNS[column] for column in benchmark_columns]]
        return benchmark

    def load_all(self) -> pd.DataFrame:
        """按基础行情、可选因子结果、benchmark 的顺序加载完整数据。"""
        self.data = self.load_basic()
        if self.factor_dir is not None:
            self.factor_result = self.load_factor_result()
        if self.constituents_path is not None and self.constituent_index is not None:
            self.constituents = self.load_constituents()
            self.data = filter_by_constituent(self.data, self.constituents)
        if self.security_status_path is not None:
            self.security_status = self.load_security_status()
            self.data = filter_by_security_status(
                self.data,
                self.security_status,
            )
        self.benchmark = self.load_benchmark()
        return self.data
