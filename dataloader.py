"""读取按股票拆分的数据和每日 S&P 500 成分股。"""

from datetime import date
from pathlib import Path
import pandas as pd


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
        csv_files = sorted(data_dir.glob("*.csv"))
    else:
        csv_files = [data_dir / f"{symbol}.csv" for symbol in symbols]

    frames = []
    for csv_file in csv_files:
        usecols = ["date", *columns] if columns is not None else None
        frame = pd.read_csv(csv_file, usecols=usecols, parse_dates=["date"])
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
    ) -> None:
        self.data_dir = data_dir
        self.symbols = symbols
        self.start = start
        self.end = end
        self.columns = columns
        self.factor_dir = factor_dir
        self.constituents_path = constituents_path
        self.constituent_index = constituent_index
        if (constituents_path is None) != (constituent_index is None):
            raise ValueError("constituents_path 和 constituent_index 必须同时为 None 或同时不为 None")
        self.data = pd.DataFrame()
        self.factor_result = pd.DataFrame()
        self.benchmark = pd.DataFrame()
        self.constituents = pd.DataFrame()

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

        constituents = pd.read_csv(
            self.constituents_path,
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

        if not self.data.empty:
            data_dates = pd.DatetimeIndex(
                self.data.index.get_level_values("date").unique()
            ).sort_values()
            target_dates = data_dates[data_dates >= constituents.index.min()]
            if target_dates.empty:
                raise ValueError(
                    f"{self.constituent_index} 的成分股日期与行情日期没有交集"
                )
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
        """提取回测期间的 HS300 收盘价。"""
        if self.data.empty:
            raise ValueError("加载 benchmark 前必须先加载行情数据")
        source_symbol = self.data.index.get_level_values("symbol")[0]
        benchmark_data = load_data(
            data_dir=self.data_dir,
            symbols=[source_symbol],
            start=self.start,
            end=self.end,
            columns=["HS300_close"],
        )
        benchmark = benchmark_data.xs(source_symbol, level="symbol").rename(
            columns={"HS300_close": "HS300"}
        )[["HS300"]]
        return benchmark

    def load_all(self) -> pd.DataFrame:
        """按基础行情、可选因子结果、benchmark 的顺序加载完整数据。"""
        self.data = self.load_basic()
        if self.factor_dir is not None:
            self.factor_result = self.load_factor_result()
        if self.constituents_path is not None and self.constituent_index is not None:
            self.constituents = self.load_constituents()
            self.data = filter_by_constituent(self.data, self.constituents)
        self.benchmark = self.load_benchmark()
        return self.data
