"""读取按股票拆分的数据、指数成分股和基准数据。"""

from datetime import date
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

def _read_header(csv_file: Path) -> set[str]:
    """读取 CSV 表头。"""
    return set(pd.read_csv(csv_file, nrows=0).columns)


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

    if symbols is None:
        csv_files = []
        for csv_file in data_dir.glob("*.csv"):
            if csv_file.name in NON_STOCK_CSV_FILES:
                continue
            csv_files.append(csv_file)
        csv_files = sorted(csv_files)
    else:
        csv_files = [data_dir / f"{symbol}.csv" for symbol in symbols]

    frames = []
    for csv_file in csv_files:
        available_columns = _read_header(csv_file)
        date_column = resolve_date_column(available_columns, csv_file)
        usecols = resolve_source_columns(columns, available_columns, date_column, csv_file)
        frame = pd.read_csv(csv_file, usecols=usecols, parse_dates=[date_column])
        frame = normalize_fields(frame, columns)
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
        self.security_status = pd.DataFrame()

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
            self.security_status = self.load_security_status()
            self.data = filter_by_security_status(
                self.data,
                self.security_status,
            )
        self.benchmark = self.load_benchmark()
        return self.data
