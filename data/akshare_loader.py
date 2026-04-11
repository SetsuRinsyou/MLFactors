"""AKShare 数据加载器。

需要安装可选依赖:  pip install akshare
"""

from __future__ import annotations

import hashlib
import time
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from config import get_config
from data.base import DataLoader
from data.schema import Col


def _ensure_akshare():
    try:
        import akshare  # noqa: F401
        return akshare
    except ImportError:
        raise ImportError("请先安装 akshare:  pip install akshare")


class AKShareLoader(DataLoader):
    """通过 AKShare 获取行情和基本面数据。

    Parameters
    ----------
    cache_dir : 本地缓存目录，None 则使用配置文件中的默认值
    request_interval : 请求间隔（秒），防止频率限制
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        request_interval: float = 0.3,
    ) -> None:
        cfg = get_config()
        self.cache_dir = Path(cache_dir or cfg["data"]["cache_dir"]).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_interval = request_interval

    # ------------------------------------------------------------------ #
    #  缓存
    # ------------------------------------------------------------------ #

    def _cache_key(self, prefix: str, **kwargs) -> str:
        raw = f"{prefix}_" + "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return hashlib.md5(raw.encode()).hexdigest()

    def _load_cache(self, key: str) -> pd.DataFrame | None:
        path = self.cache_dir / f"{key}.parquet"
        if path.exists():
            logger.debug("命中缓存: {}", path)
            return pd.read_parquet(path)
        return None

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        path = self.cache_dir / f"{key}.parquet"
        df.to_parquet(path)
        logger.debug("缓存已保存: {}", path)

    # ------------------------------------------------------------------ #
    #  DataLoader 接口
    # ------------------------------------------------------------------ #

    def load_market_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        ak = _ensure_akshare()

        if symbols is None:
            raise ValueError("AKShareLoader 需要指定 symbols 列表")

        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            cache_key = self._cache_key("market", symbol=symbol, start=str(start), end=str(end))
            cached = self._load_cache(cache_key)
            if cached is not None:
                frames.append(cached)
                continue

            logger.info("从 AKShare 获取 {} 行情数据", symbol)
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=str(start).replace("-", "") if start else "19700101",
                    end_date=str(end).replace("-", "") if end else "20991231",
                    adjust="qfq",
                )
            except Exception as e:
                logger.warning("获取 {} 失败: {}", symbol, e)
                continue

            # 标准化列名
            rename_map = {
                "日期": Col.DATE, "股票代码": Col.SYMBOL,
                "开盘": Col.OPEN, "最高": Col.HIGH, "最低": Col.LOW,
                "收盘": Col.CLOSE, "成交量": Col.VOLUME, "成交额": Col.AMOUNT,
                "换手率": Col.TURNOVER,
            }
            df = df.rename(columns=rename_map)
            if Col.SYMBOL not in df.columns:
                df[Col.SYMBOL] = symbol

            self._save_cache(cache_key, df)
            frames.append(df)
            time.sleep(self.request_interval)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result = self._standardize(result)
        result = self._set_index(result)
        return result

    def load_fundamental_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        ak = _ensure_akshare()

        if symbols is None:
            raise ValueError("AKShareLoader 需要指定 symbols 列表")

        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            cache_key = self._cache_key("fundamental", symbol=symbol, start=str(start), end=str(end))
            cached = self._load_cache(cache_key)
            if cached is not None:
                frames.append(cached)
                continue

            logger.info("从 AKShare 获取 {} 基本面数据", symbol)
            try:
                df = ak.stock_financial_analysis_indicator(symbol=symbol)
            except Exception as e:
                logger.warning("获取 {} 基本面失败: {}", symbol, e)
                continue

            df[Col.SYMBOL] = symbol
            self._save_cache(cache_key, df)
            frames.append(df)
            time.sleep(self.request_interval)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result = self._standardize(result)
        result = self._filter(result, symbols, start, end)
        result = self._set_index(result)
        return result
