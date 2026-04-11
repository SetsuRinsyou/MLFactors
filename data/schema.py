"""统一数据列名定义。

所有 DataLoader 输出的 DataFrame 必须使用这些标准列名，
确保下游因子计算和评估模块无需关心数据来源。
"""

from __future__ import annotations


class Col:
    """行情数据标准列名。"""

    DATE = "date"
    SYMBOL = "symbol"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    AMOUNT = "amount"
    ADJ_CLOSE = "adj_close"
    RETURNS = "returns"
    VWAP = "vwap"

    # 辅助列
    TURNOVER = "turnover"
    MKT_CAP = "mkt_cap"

    @classmethod
    def market_required(cls) -> list[str]:
        return [cls.DATE, cls.SYMBOL, cls.OPEN, cls.HIGH, cls.LOW, cls.CLOSE, cls.VOLUME]

    @classmethod
    def market_all(cls) -> list[str]:
        return [
            cls.DATE, cls.SYMBOL, cls.OPEN, cls.HIGH, cls.LOW, cls.CLOSE,
            cls.VOLUME, cls.AMOUNT, cls.ADJ_CLOSE, cls.RETURNS, cls.VWAP,
            cls.TURNOVER, cls.MKT_CAP,
        ]


class FundamentalCol:
    """基本面数据标准列名。"""

    DATE = "date"
    SYMBOL = "symbol"

    # 估值
    PE = "pe"
    PE_TTM = "pe_ttm"
    PB = "pb"
    PS = "ps"
    PS_TTM = "ps_ttm"
    EV_EBITDA = "ev_ebitda"

    # 盈利
    ROE = "roe"
    ROA = "roa"
    GROSS_MARGIN = "gross_margin"
    NET_MARGIN = "net_margin"
    EPS = "eps"

    # 成长
    REVENUE_GROWTH = "revenue_growth"
    PROFIT_GROWTH = "profit_growth"

    # 杠杆
    DEBT_RATIO = "debt_ratio"
    CURRENT_RATIO = "current_ratio"

    @classmethod
    def required(cls) -> list[str]:
        return [cls.DATE, cls.SYMBOL]
