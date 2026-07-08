"""读取按股票拆分的数据、指数成分股和基准数据。"""

from datetime import date
from pathlib import Path
import pandas as pd


DATE_COLUMN_CANDIDATES = ("date", "trade_date")

FIELD_ALIASES = {
    "adj_close": ("adj_close", "close"),
    "gross_margin": ("grossprofit_margin", "gross_margin_calc"),
    "operating_cash_flow": (
        "operating_cashflow",
        "n_cashflow_act",
        "im_net_cashflow_oper_act",
    ),
    "share_factor": ("adj_factor",),
    "sector": ("industry_1", "stock_basic_industry", "sector"),
    "SPY_close": ("hs300_close",),
    "QQQ_close": ("zz500_close",),
    "HS300_close": ("hs300_close",),
    "ZZ500_close": ("zz500_close",),
    "ZZ1000_close": ("zz1000_close",),
}

DERIVED_FIELD_SOURCES = {
    "gross_margin": ("grossprofit_margin", "gross_margin_calc", "gross_profit", "revenue", "cost_revenue"),
    "gross_profit": ("revenue", "cost_revenue"),
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

NON_STOCK_CSV_FILES = {
    "constituents_daily.csv",
    "index_members_rebalance.csv",
}

SECURITY_STATUS_COLUMNS = [
    "record_type",
    "ticker",
    "trade_date",
    "start_date",
    "end_date",
    "is_suspended",
    "is_st",
    "listing_status",
]

_SECURITY_STATUS_CACHE: dict[Path, pd.DataFrame] = {}


def _read_header(csv_file: Path) -> set[str]:
    """读取 CSV 表头。"""
    return set(pd.read_csv(csv_file, nrows=0).columns)


def _list_stock_csv_files(data_dir: Path) -> list[Path]:
    """列出按股票拆分的数据文件，跳过指数成分股等辅助 CSV。"""
    return sorted(
        csv_file
        for csv_file in data_dir.glob("*.csv")
        if csv_file.name not in NON_STOCK_CSV_FILES
    )


def _resolve_date_column(columns: set[str], csv_file: Path) -> str:
    """返回数据文件实际使用的日期列名。"""
    for column in DATE_COLUMN_CANDIDATES:
        if column in columns:
            return column
    raise ValueError(f"{csv_file} 缺少日期列，支持字段: {DATE_COLUMN_CANDIDATES}")


def _source_columns_for_field(field: str, available: set[str]) -> list[str]:
    """把统一字段名解析为当前 CSV 需要读取的源字段。"""
    if field == "date":
        return []
    if field in DERIVED_FIELD_SOURCES and field not in available:
        sources = [column for column in DERIVED_FIELD_SOURCES[field] if column in available]
        if sources:
            return sources
    candidates = (*FIELD_ALIASES.get(field, ()), field)
    for candidate in candidates:
        if candidate in available:
            return [candidate]
    return []


def _resolve_source_columns(
    requested_columns: list[str] | None,
    available: set[str],
    date_column: str,
    csv_file: Path,
) -> list[str] | None:
    """构造 ``read_csv`` 的 usecols，并在字段缺失时给出明确错误。"""
    if requested_columns is None:
        return None

    source_columns = [date_column]
    missing = []
    for field in dict.fromkeys(requested_columns):
        resolved = _source_columns_for_field(field, available)
        if not resolved:
            missing.append(field)
            continue
        source_columns.extend(resolved)
    if missing:
        raise ValueError(f"{csv_file} 缺少字段: {missing}")
    return list(dict.fromkeys(source_columns))


def _coalesce_columns(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    """按候选顺序返回逐列回退后的 Series。"""
    sources = [column for column in candidates if column in frame.columns]
    if not sources:
        return None
    if len(sources) == 1:
        return frame[sources[0]]
    return frame[sources].bfill(axis=1).iloc[:, 0]


def _normalize_fields(frame: pd.DataFrame, requested_columns: list[str] | None) -> pd.DataFrame:
    """把 A 股源字段规范为因子代码使用的统一字段名。"""
    if "trade_date" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"trade_date": "date"})

    target_columns = set(requested_columns or frame.columns)
    for target, candidates in FIELD_ALIASES.items():
        if target not in target_columns and requested_columns is not None:
            continue
        values = _coalesce_columns(frame, candidates)
        if values is not None:
            frame[target] = values

    if (
        ("gross_margin" in target_columns or requested_columns is None)
        and {"gross_profit", "revenue"}.issubset(frame.columns)
    ):
        calculated = frame["gross_profit"] / frame["revenue"].replace(0, pd.NA)
        if "gross_margin" in frame.columns:
            frame["gross_margin"] = frame["gross_margin"].fillna(calculated)
        else:
            frame["gross_margin"] = calculated

    if (
        ("gross_profit" in target_columns or requested_columns is None)
        and "gross_profit" not in frame.columns
        and {"revenue", "cost_revenue"}.issubset(frame.columns)
    ):
        frame["gross_profit"] = frame["revenue"] - frame["cost_revenue"]

    debt_sources = [column for column in DERIVED_FIELD_SOURCES["non_current_debt"] if column in frame.columns]
    if (
        ("non_current_debt" in target_columns or requested_columns is None)
        and "non_current_debt" not in frame.columns
        and debt_sources
    ):
        frame["non_current_debt"] = frame[debt_sources].fillna(0).sum(axis=1, min_count=1)

    if requested_columns is None:
        return frame

    output_columns = ["date", *dict.fromkeys(requested_columns)]
    return frame[[column for column in output_columns if column in frame.columns]]


def load_data(
    data_dir: str | Path,
    symbols: list[str] | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """返回 (date, symbol) MultiIndex 数据"""
    data_dir = Path(data_dir).expanduser().resolve()

    if symbols is None:
        csv_files = _list_stock_csv_files(data_dir)
    else:
        csv_files = [data_dir / f"{symbol}.csv" for symbol in symbols]

    frames = []
    for csv_file in csv_files:
        available_columns = _read_header(csv_file)
        date_column = _resolve_date_column(available_columns, csv_file)
        usecols = _resolve_source_columns(columns, available_columns, date_column, csv_file)
        frame = pd.read_csv(csv_file, usecols=usecols, parse_dates=[date_column])
        frame = _normalize_fields(frame, columns)
        frame["symbol"] = csv_file.stem
        if start is not None:
            frame = frame[frame["date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame["date"] <= pd.Timestamp(end)]
        frames.append(frame)

    if frames:
        data = pd.concat(frames, ignore_index=True)
        data = data.set_index(["date", "symbol"]).sort_index()
    else:
        index = pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
        data = pd.DataFrame(index=index)

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


def _truthy(series: pd.Series) -> pd.Series:
    """将 tushare 状态字段转换为布尔值。"""
    return series.astype(str).str.lower().isin({"t", "true", "1", "y", "yes"})


def _build_security_exclusion_index(
    status: pd.DataFrame,
    data_index: pd.MultiIndex,
) -> pd.MultiIndex:
    """根据 ST、停牌、退市记录构造需要剔除的 (date, symbol) 索引。"""
    if data_index.empty or status.empty:
        return pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])

    data_dates = pd.DatetimeIndex(data_index.get_level_values("date").unique()).sort_values()
    data_symbols = pd.Index(data_index.get_level_values("symbol").unique())
    if data_dates.empty or data_symbols.empty:
        return pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])

    status = status[status["ticker"].isin(data_symbols)].copy()
    if status.empty:
        return pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])

    excluded_dates = []
    excluded_symbols = []

    suspended = status[
        status["record_type"].eq("suspend_daily")
        & _truthy(status["is_suspended"])
    ].copy()
    if not suspended.empty:
        suspended["date"] = pd.to_datetime(suspended["trade_date"], errors="coerce")
        suspended = suspended[
            suspended["date"].isin(data_dates)
            & suspended["ticker"].isin(data_symbols)
        ]
        if not suspended.empty:
            excluded_dates.append(suspended["date"].to_numpy())
            excluded_symbols.append(suspended["ticker"].to_numpy())

    interval_filters = [
        status["record_type"].eq("st_status_history") & _truthy(status["is_st"]),
        status["record_type"].eq("listing_status_history")
        & status["listing_status"].astype(str).str.lower().eq("delisted"),
    ]
    for mask in interval_filters:
        intervals = status[mask].copy()
        if intervals.empty:
            continue
        intervals["start"] = pd.to_datetime(intervals["start_date"], errors="coerce")
        intervals["end"] = pd.to_datetime(intervals["end_date"], errors="coerce").fillna(data_dates[-1])
        intervals = intervals.dropna(subset=["start"])
        for row in intervals.itertuples(index=False):
            left = data_dates.searchsorted(row.start, side="left")
            right = data_dates.searchsorted(row.end, side="right")
            if right <= left:
                continue
            dates = data_dates[left:right]
            excluded_dates.append(dates.to_numpy())
            excluded_symbols.append(pd.Index([row.ticker] * len(dates)).to_numpy())

    if not excluded_dates:
        return pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])

    return pd.MultiIndex.from_arrays(
        [pd.Index(pd.concat([pd.Series(values) for values in excluded_dates])).to_numpy(),
         pd.Index(pd.concat([pd.Series(values) for values in excluded_symbols])).to_numpy()],
        names=["date", "symbol"],
    ).drop_duplicates()


def filter_by_security_status(
    data: pd.DataFrame,
    security_status_path: str | Path,
) -> tuple[pd.DataFrame, pd.MultiIndex]:
    """剔除每日 ST、停牌、退市股票。"""
    if data.empty:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
        return data, empty_index
    if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "symbol"]:
        raise ValueError("data 必须使用 (date, symbol) MultiIndex")

    security_status_path = Path(security_status_path)
    if not security_status_path.exists():
        raise FileNotFoundError(f"证券状态文件不存在: {security_status_path}")

    security_status_path = security_status_path.resolve()
    if security_status_path not in _SECURITY_STATUS_CACHE:
        _SECURITY_STATUS_CACHE[security_status_path] = pd.read_csv(
            security_status_path,
            usecols=SECURITY_STATUS_COLUMNS,
            dtype=str,
        )
    status = _SECURITY_STATUS_CACHE[security_status_path]
    excluded_index = _build_security_exclusion_index(status, data.index)
    if excluded_index.empty:
        return data, excluded_index
    return data.loc[~data.index.isin(excluded_index)].sort_index(), excluded_index


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
        self.constituents_path = constituents_path
        self.constituent_index = constituent_index
        self.security_status_path = security_status_path
        if (constituents_path is None) != (constituent_index is None):
            raise ValueError("constituents_path 和 constituent_index 必须同时为 None 或同时不为 None")
        self.data = pd.DataFrame()
        self.factor_result = pd.DataFrame()
        self.benchmark = pd.DataFrame()
        self.constituents = pd.DataFrame()
        self.security_exclusions = pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])

    def load_basic(self) -> pd.DataFrame:
        """加载基础行情数据和可选成分股。"""
        basic_data = load_data(
            data_dir=self.data_dir,
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            columns=self.columns,
        )
        return basic_data

    def load_constituents(self) -> pd.DataFrame:
        """加载成分股数据。"""

        if self.constituents_path is None:
            return pd.DataFrame()
        constituents_path = Path(self.constituents_path)
        if not constituents_path.exists():
            raise FileNotFoundError(f"成分股文件不存在: {constituents_path}")

        header = _read_header(constituents_path)
        if {"date", "tickers"}.issubset(header):
            constituents_daily = pd.read_csv(
                constituents_path,
                usecols=["date", "tickers"],
                parse_dates=["date"],
            )
            records = []
            for row in constituents_daily.itertuples(index=False):
                tickers = str(row.tickers).split(",") if pd.notna(row.tickers) else []
                records.extend((row.date, ticker.strip(), True) for ticker in tickers if ticker.strip())
            constituents = pd.DataFrame(records, columns=["date", "symbol", "is_member"])
            if constituents.empty:
                raise ValueError(f"{constituents_path} 没有可用成分股数据")
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
            return self._align_constituents_to_data(constituents)

        constituents = pd.read_csv(
            constituents_path,
            dtype={
                "index_code": str,
                "con_code": str,
            },
            usecols=["index_code", "con_code", "trade_date"],
            parse_dates=["trade_date"],
        )
        # 仅保留指定指数的成分股
        constituents = constituents[constituents["index_code"] == self.constituent_index]
        # 仅保留回测结束日前的成分股；开始日前的最后一条记录用于对齐首个行情日。
        if self.data.empty and self.start is not None:
            constituents = constituents[constituents["trade_date"] >= pd.Timestamp(self.start)]
        if self.end is not None:
            constituents = constituents[constituents["trade_date"] <= pd.Timestamp(self.end)]

        if constituents.empty:
            raise ValueError(f"在 {self.start} 到 {self.end} 期间没有找到 {self.constituent_index} 的成分股数据")

        constituents["is_member"] = 1
        constituents = (
            constituents.pivot_table(
                index="trade_date",
                columns="con_code",
                values="is_member",
                aggfunc="last",
                fill_value=0,
            )
            .astype(bool)
            .sort_index()
            .sort_index(axis=1)
        )
        constituents.index.name = "date"
        constituents.columns.name = "symbol"

        return self._align_constituents_to_data(constituents)

    def _align_constituents_to_data(self, constituents: pd.DataFrame) -> pd.DataFrame:
        """将成分股矩阵前向填充到行情交易日。"""
        if self.data.empty:
            return constituents

        data_dates = pd.DatetimeIndex(
            self.data.index.get_level_values("date").unique()
        ).sort_values()
        target_dates = data_dates[data_dates >= constituents.index.min()]
        if target_dates.empty:
            raise ValueError(f"{self.constituent_index} 的成分股日期与行情日期没有交集")

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

    def load_factor_result(self) -> pd.DataFrame:
        """加载已保存的因子宽表，并合并到基础行情数据。"""

        factor_result = load_data(
            data_dir=self.factor_dir,
            symbols=self.symbols,
            start=self.start,
            end=self.end,
        )
        if not self.data.empty:
            self.data = self.data.join(factor_result, how="left")
        return factor_result

    def load_benchmark(self) -> pd.DataFrame:
        """提取回测期间可用的指数基准收盘价。"""
        if self.data.empty:
            return pd.DataFrame()
        source_symbol = self.data.index.get_level_values("symbol")[0]
        source_file = Path(self.data_dir).expanduser().resolve() / f"{source_symbol}.csv"
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
            self.data, self.security_exclusions = filter_by_security_status(
                self.data,
                self.security_status_path,
            )
        self.benchmark = self.load_benchmark()
        return self.data
